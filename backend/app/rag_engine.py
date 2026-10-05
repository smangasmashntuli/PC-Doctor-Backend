#rag_engine.py
import requests
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import urljoin, urlparse, parse_qs
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from backend.app.vector_store import get_vector_store
from backend.app.models import LaptopSpecs, LaptopSetup
from backend.app.database import SessionLocal

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def ingest_laptop_specs(self, brand: str, model_name: str, user_id: int) -> Dict:
        """
        Main entry point: Fetch laptop specs from web sources and store in vector DB
        
        Args:
            brand: Laptop brand (e.g., "Dell", "Lenovo")
            model_name: Model name/number (e.g., "XPS 15 9520")
            user_id: User ID for database association
        
        Returns:
            Dict with ingestion results
        """
        try:
            # Step 1: Search for official documentation
            search_results = await self._search_laptop_documentation(brand, model_name)
            
            # Step 2: Scrape and extract specs from top results
            scraped_data = await self._scrape_specs_from_sources(search_results)
            
            # Step 3: Parse and structure the specs
            structured_specs = self._parse_specs(scraped_data, brand, model_name)

            # Regex extraction is intentionally conservative; use Gemini to fill
            # fields that source pages do not expose as plain text.
            core_fields = ("cpu", "gpu", "ram", "storage", "display", "os")
            if sum(bool(structured_specs.get(field)) for field in core_fields) < 2:
                try:
                    from backend.app.gemini_service import get_gemini_service

                    enriched_specs = await get_gemini_service().enrich_laptop_specs(
                        brand=brand,
                        model_name=model_name,
                        current_specs=structured_specs,
                    )
                    for field, value in enriched_specs.items():
                        if not structured_specs.get(field):
                            structured_specs[field] = value
                except Exception as enrichment_error:
                    logger.warning(f"Gemini laptop spec enrichment unavailable: {enrichment_error}")
            
            # Step 4: Store in vector database
            sources_found = self._store_in_vector_db(structured_specs, brand, model_name)
            
            # Step 5: Save to MySQL
            db = SessionLocal()
            try:
                # Get or create the user's single laptop setup
                laptop_setup = db.query(LaptopSetup).filter(
                    LaptopSetup.user_id == user_id
                ).first()
                
                if not laptop_setup:
                    laptop_setup = LaptopSetup(
                        user_id=user_id,
                        brand=brand,
                        model=model_name
                    )
                    db.add(laptop_setup)
                    db.commit()
                    db.refresh(laptop_setup)
                elif laptop_setup.brand != brand or laptop_setup.model != model_name:
                    laptop_setup.brand = brand
                    laptop_setup.model = model_name
                    db.commit()
                    db.refresh(laptop_setup)
                
                # Create or update laptop specs
                specs = db.query(LaptopSpecs).filter(
                    LaptopSpecs.laptop_setup_id == laptop_setup.id
                ).first()
                
                if not specs:
                    specs = LaptopSpecs(laptop_setup_id=laptop_setup.id)
                    db.add(specs)
                
                # Update specs
                specs.cpu = structured_specs.get('cpu')
                specs.gpu = structured_specs.get('gpu')
                specs.ram = structured_specs.get('ram')
                specs.storage = structured_specs.get('storage')
                specs.display = structured_specs.get('display')
                specs.ports = json.dumps(structured_specs.get('ports', []))
                specs.os = structured_specs.get('os')
                specs.known_issues = json.dumps(structured_specs.get('known_issues', []))
                # Only overwrite the stored image when scraping actually found
                # a new one. Assigning the (usually None) scraped value here
                # wiped a previously generated image on every re-ingest, which
                # is why the UI stopped showing it.
                scraped_image_url = structured_specs.get('image_url')
                if scraped_image_url:
                    specs.image_url = scraped_image_url
                specs.source_urls = json.dumps(structured_specs.get('source_urls', []))
                specs.updated_at = datetime.utcnow()
                
                db.commit()
                db.refresh(specs)
                
                return {
                    "message": "Laptop specs ingested successfully",
                    "laptop_setup_id": laptop_setup.id,
                    "specs": specs,
                    "sources_found": sources_found
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error ingesting laptop specs: {e}")
            raise
    
    async def _search_laptop_documentation(self, brand: str, model_name: str) -> List[Dict]:
        """Search for official documentation and spec sheets"""
        search_queries = [
            f"{brand} {model_name} specifications",
            f"{brand} {model_name} manual pdf",
            f"{brand} {model_name} tech specs",
            f"{brand} {model_name} review specs"
        ]
        
        results = []
        
        # DuckDuckGo is used because it needs no API key. It intermittently
        # answers HTTP 202 (bot challenge) with an empty result set, so we try
        # both endpoints and both the dedicated and generic anchor selectors.
        endpoints = (
            "https://html.duckduckgo.com/html/",
            "https://lite.duckduckgo.com/lite/",
        )

        for query in search_queries[:2]:  # Limit to 2 queries to avoid rate limits
            for endpoint in endpoints:
                if len(results) >= 5:
                    break
                try:
                    response = requests.get(
                        endpoint,
                        params={"q": query},
                        headers=self.headers,
                        timeout=10,
                    )
                    status_code = getattr(response, "status_code", None)
                    # Only reject when we can positively identify a failure;
                    # a non-integer value means the response is not a real
                    # HTTP result (e.g. a test double) so we parse it anyway.
                    if isinstance(status_code, int) and status_code != 200:
                        logger.warning(
                            f"Search endpoint {endpoint} returned HTTP {status_code}"
                        )
                        continue

                    soup = BeautifulSoup(response.text, 'html.parser')
                    anchors = soup.find_all('a', class_='result__a', limit=5) or \
                        soup.find_all('a', class_='result-link', limit=5)
                    if not anchors:
                        # Generic fallback: any absolute http(s) link that is
                        # not itself a search-engine page.
                        anchors = [
                            a for a in soup.find_all('a', href=True)
                            if a['href'].startswith('http')
                        ][:5]

                    for anchor in anchors:
                        if len(results) >= 5:
                            break
                        href = self._normalize_result_url(anchor.get('href', ''))
                        title = anchor.get_text(strip=True)
                        if not href or not title:
                            continue
                        results.append({
                            'url': href,
                            'title': title,
                            'source': 'duckduckgo'
                        })
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}' at {endpoint}: {e}")

        # De-duplicate while preserving order
        seen = set()
        deduped = []
        for item in results:
            if item['url'] in seen:
                continue
            seen.add(item['url'])
            deduped.append(item)
        results = deduped
        
        # Prioritize official sources
        official_domains = [f'{brand.lower()}.com', f'{brand.lower()}.co.uk']
        prioritized_results = sorted(
            results,
            key=lambda x: any(domain in x['url'].lower() for domain in official_domains),
            reverse=True
        )
        
        return prioritized_results[:5]  # Return top 5 results

    @staticmethod
    def _normalize_result_url(href: str) -> Optional[str]:
        """Resolve a search-result href into a direct http(s) URL.

        DuckDuckGo wraps outbound links in ``/l/?uddg=<encoded>`` redirects,
        and returns protocol-relative links on the lite endpoint.
        """
        if not href:
            return None

        href = href.strip()
        if href.startswith('//'):
            href = 'https:' + href

        parsed = urlparse(href)
        if parsed.netloc.endswith('duckduckgo.com') and parsed.path.startswith('/l/'):
            target = parse_qs(parsed.query).get('uddg', [None])[0]
            if target:
                href = target

        if not href.startswith(('http://', 'https://')):
            return None

        host = urlparse(href).netloc.lower()
        if 'duckduckgo.com' in host:
            return None
        return href

    async def _scrape_specs_from_sources(self, sources: List[Dict]) -> List[Dict]:
        """Scrape laptop specs from source URLs"""
        scraped_data = []
        
        for source in sources[:3]:  # Limit to top 3 sources
            try:
                response = requests.get(source['url'], headers=self.headers, timeout=15)
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Extract text content
                text_content = soup.get_text(separator=' ', strip=True)
                
                # Remove excessive whitespace
                text_content = re.sub(r'\s+', ' ', text_content)

                image_url = None
                image_tag = (
                    soup.find('meta', property='og:image')
                    or soup.find('meta', attrs={'name': 'twitter:image'})
                )
                if image_tag and image_tag.get('content'):
                    image_url = urljoin(source['url'], image_tag['content'])
                
                scraped_data.append({
                    'url': source['url'],
                    'title': source['title'],
                    'content': text_content[:5000],  # Limit content size
                    'image_url': image_url,
                    'scraped_at': datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.warning(f"Failed to scrape {source['url']}: {e}")
                continue
        
        return scraped_data
    
    def _parse_specs(self, scraped_data: List[Dict], brand: str, model_name: str) -> Dict:
        """Parse structured specs from scraped text using regex patterns"""
        combined_text = ' '.join([d['content'] for d in scraped_data])
        
        specs = {
            'brand': brand,
            'model': model_name,
            'cpu': self._extract_cpu(combined_text),
            'gpu': self._extract_gpu(combined_text),
            'ram': self._extract_ram(combined_text),
            'storage': self._extract_storage(combined_text),
            'display': self._extract_display(combined_text),
            'ports': self._extract_ports(combined_text),
            'os': self._extract_os(combined_text),
            'known_issues': self._extract_known_issues(combined_text),
            'image_url': next((item.get('image_url') for item in scraped_data if item.get('image_url')), None),
            'source_urls': list(set([d['url'] for d in scraped_data]))
        }
        
        return specs
    
    def _extract_cpu(self, text: str) -> Optional[str]:
        """Extract CPU information"""
        patterns = [
            r'(Intel\s+Core\s+i[3579]-\d+\w*(?:\s+Processor)?)',
            r'(AMD\s+Ryzen\s+[3579]\s+\d+\w*)',
            r'(Intel\s+Core\s+i[3579]\s+\d+\w*)',
            r'(Apple\s+M[123]\s+chip)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_gpu(self, text: str) -> Optional[str]:
        """Extract GPU information"""
        patterns = [
            r'(NVIDIA\s+(?:GeForce\s+)?RTX\s+\d+\w*)',
            r'(NVIDIA\s+(?:GeForce\s+)?GTX\s+\d+\w*)',
            r'(AMD\s+Radeon\s+\w+)',
            r'(Intel\s+UHD\s+Graphics\s+\d+)',
            r'(Intel\s+Iris\s+Xe\s+Graphics)',
            r'(Apple\s+GPU)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_ram(self, text: str) -> Optional[str]:
        """Extract RAM information"""
        patterns = [
            r'(\d+\s*GB\s+(?:DDR[45]|LPDDR[45]|DDR5)\s+RAM)',
            r'(\d+\s*GB\s+(?:DDR[45]|LPDDR[45]))',
            r'(\d+\s*GB\s+memory)',
            r'(\d+\s*GB\s+RAM)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_storage(self, text: str) -> Optional[str]:
        """Extract storage information"""
        patterns = [
            r'(\d+\s*(?:GB|TB)\s+(?:SSD|NVMe|PCIe|storage))',
            r'(\d+\s*(?:GB|TB)\s+(?:solid state drive|hard drive))',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1)
                if 'nvme' in value.lower() and 'ssd' not in value.lower():
                    return f'{value} SSD'
                return value
        return None
    
    def _extract_display(self, text: str) -> Optional[str]:
        """Extract display information"""
        patterns = [
            r'(\d+\.?\d*"\s+(?:FHD|UHD|4K|HD|IPS|OLED).*?display)',
            r'(\d+\.?\d*"\s+(?:FHD|UHD|4K|HD|IPS|OLED))',
            r'(\d+\.?\d*\s*inch\s+.*?display)',
            r'(\d+\s*x\s*\d+\s+resolution)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1)[:100]  # Limit length
        return None
    
    def _extract_ports(self, text: str) -> List[str]:
        """Extract port information"""
        ports = []
        port_patterns = [
            r'USB-C\s*(?:Thunderbolt)?',
            r'USB-A\s*\d+\.?\d*',
            r'HDMI\s*\d+\.?\d*',
            r'SD\s*card\s*reader',
            r'3\.5mm\s*audio\s*jack',
            r'Ethernet',
            r'Thunderbolt\s*\d+',
        ]
        
        for pattern in port_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            ports.extend([m.strip() for m in matches])
        
        return list(set(ports))[:10]  # Return unique ports, max 10
    
    def _extract_os(self, text: str) -> Optional[str]:
        """Extract operating system information"""
        patterns = [
            r'(Windows\s+\d+\s*(?:Pro|Home)?)',
            r'(macOS\s+\w+)',
            r'(Linux\s+\w+)',
            r'(Chrome\s*OS)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_known_issues(self, text: str) -> List[str]:
        """Extract known issues from reviews and forums"""
        issues = []
        issue_keywords = [
            'problem', 'issue', 'defect', 'recall', 'battery drain',
            'overheating', 'fan noise', 'display flicker', 'wifi issue',
            'bluetooth issue', 'keyboard problem', 'touchpad issue'
        ]
        
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in issue_keywords):
                issue = sentence.strip()
                if 20 < len(issue) < 200:  # Reasonable length
                    issues.append(issue)
        
        return issues[:5]  # Return top 5 issues
    
    def _store_in_vector_db(self, specs: Dict, brand: str, model_name: str) -> int:
        """Store specs as documentation chunks in ChromaDB"""
        chunks_stored = 0
        
        # Create chunks for different spec sections
        spec_chunks = [
            {
                'section': 'overview',
                'text': f"{brand} {model_name} - {specs.get('cpu', 'N/A')} processor, {specs.get('ram', 'N/A')} RAM, {specs.get('storage', 'N/A')} storage.",
            },
            {
                'section': 'processor',
                'text': f"CPU: {specs.get('cpu', 'Not specified')}",
            },
            {
                'section': 'memory',
                'text': f"RAM: {specs.get('ram', 'Not specified')}",
            },
            {
                'section': 'storage',
                'text': f"Storage: {specs.get('storage', 'Not specified')}",
            },
            {
                'section': 'display',
                'text': f"Display: {specs.get('display', 'Not specified')}",
            },
            {
                'section': 'ports',
                'text': f"Ports: {', '.join(specs.get('ports', [])) or 'Not specified'}",
            },
            {
                'section': 'known_issues',
                'text': f"Known issues: {'; '.join(specs.get('known_issues', [])) or 'None reported'}",
            }
        ]
        
        for chunk in spec_chunks:
            chunk_id = f"{brand}_{model_name}_{chunk['section']}_{datetime.utcnow().timestamp()}"
            metadata = {
                'chunk_id': chunk_id,
                'brand': brand,
                'model': model_name,
                'section': chunk['section'],
                'source_urls': json.dumps(specs.get('source_urls', [])),
                'created_at': datetime.utcnow().isoformat()
            }
            
            if self.vector_store.add_documentation_chunk(
                chunk_id=chunk_id,
                text=chunk['text'],
                metadata=metadata
            ):
                chunks_stored += 1
        
        return chunks_stored
    
    def get_laptop_context(self, brand: str, model_name: str, query: str = "") -> str:
        """
        Retrieve relevant context for a laptop model from vector DB
        
        Args:
            brand: Laptop brand
            model_name: Model name
            query: Optional query to filter relevant chunks
        
        Returns:
            Formatted context string for AI prompt
        """
        chunks = self.vector_store.get_chunks_by_laptop(brand, model_name, n_results=5)
        
        if not chunks:
            return f"No specific documentation found for {brand} {model_name}."
        
        context_parts = [f"Documentation for {brand} {model_name}:\n"]
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"{i}. [{chunk['metadata']['section'].upper()}] {chunk['text']}")
            if chunk['metadata'].get('source_urls'):
                source_urls = json.loads(chunk['metadata']['source_urls'])
                if source_urls:
                    context_parts.append(f"   Source: {source_urls[0]}")
        
        return "\n".join(context_parts)


# Global instance
_rag_engine_instance = None

def get_rag_engine() -> RAGEngine:
    """Get or create the global RAG engine instance"""
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()
    return _rag_engine_instance
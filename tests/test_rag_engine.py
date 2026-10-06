#test_rag_engine.py
import pytest
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock chromadb before importing our modules
sys.modules['chromadb'] = Mock()
sys.modules['chromadb.config'] = Mock()

from backend.app.rag_engine import RAGEngine
from backend.app.vector_store import VectorStore
from backend.app.models import LaptopSpecs, LaptopSetup, Users
from backend.app.fetch import LaptopIngestResponse


class TestRAGEngine:
    """Test suite for Phase 1: RAG Engine & Laptop Spec Retrieval"""
    
    @pytest.fixture
    def rag_engine(self):
        """Create RAGEngine instance with mocked vector store"""
        with patch('backend.app.rag_engine.get_vector_store') as mock_vs:
            mock_vector_store = Mock(spec=VectorStore)
            mock_vs.return_value = mock_vector_store
            
            engine = RAGEngine()
            engine.vector_store = mock_vector_store
            return engine
    
    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        mock_db = Mock()
        return mock_db
    
    @pytest.mark.asyncio
    async def test_ingest_laptop_specs_dell_xps(self, rag_engine):
        """
        Test Phase 1 Requirement: Ingest laptop specs for "Dell XPS 15 9520"
        Verifies specs are saved in MySQL and context embeddings with source URLs are stored in ChromaDB
        """
        # Mock the search and scrape methods
        with patch.object(rag_engine, '_search_laptop_documentation') as mock_search, \
             patch.object(rag_engine, '_scrape_specs_from_sources') as mock_scrape, \
             patch('backend.app.rag_engine.SessionLocal') as mock_db_class:
            
            # Setup mock search results
            mock_search.return_value = [
                {
                    'url': 'https://www.dell.com/support/manual/en-us/xps-15-9520-laptop/specifications',
                    'title': 'Dell XPS 15 9520 Specifications',
                    'source': 'duckduckgo'
                }
            ]
            
            # Setup mock scraped data
            mock_scrape.return_value = [
                {
                    'url': 'https://www.dell.com/support/manual/en-us/xps-15-9520-laptop/specifications',
                    'title': 'Dell XPS 15 9520 Specifications',
                    'content': 'Intel Core i7-12700H, 16GB DDR5 RAM, 512GB SSD, NVIDIA RTX 3050 Ti, 15.6" FHD display, Windows 11 Pro',
                    'scraped_at': datetime.utcnow().isoformat()
                }
            ]
            
            # Setup mock database
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # The code issues exactly two lookups against the session:
            #   db.query(LaptopSetup).filter(...).first()  -> None (create)
            #   db.query(LaptopSpecs).filter(...).first()  -> None (create)
            # Wire the chain the way it is actually traversed:
            # db.query(...).filter(...).first()
            mock_db.query.return_value.filter.return_value.first.side_effect = [
                None,  # No existing laptop setup
                None,  # No existing specs
            ]

            # A real session.refresh() repopulates primary keys after commit;
            # a bare Mock does not, so emulate it. Without this the returned
            # laptop_setup_id stays a Mock and the assertion below fails.
            def _fake_refresh(obj):
                if isinstance(obj, LaptopSetup):
                    obj.id = 1
                elif isinstance(obj, LaptopSpecs):
                    obj.id = 1

            mock_db.refresh.side_effect = _fake_refresh
            
            # Execute ingestion
            result = await rag_engine.ingest_laptop_specs(
                brand="Dell",
                model_name="XPS 15 9520",
                user_id=1
            )
            
            # Assertions
            assert result["message"] == "Laptop specs ingested successfully"
            assert result["laptop_setup_id"] == 1
            assert result["sources_found"] > 0
            
            # Verify vector store was called
            assert rag_engine.vector_store.add_documentation_chunk.called
            
            # Verify database operations
            assert mock_db.add.called
            assert mock_db.commit.called
            assert mock_db.refresh.called
    
    @pytest.mark.asyncio
    async def test_ingest_laptop_specs_with_existing_setup(self, rag_engine):
        """Test ingestion when laptop setup already exists"""
        with patch.object(rag_engine, '_search_laptop_documentation') as mock_search, \
             patch.object(rag_engine, '_scrape_specs_from_sources') as mock_scrape, \
             patch('backend.app.rag_engine.SessionLocal') as mock_db_class:
            
            mock_search.return_value = []
            mock_scrape.return_value = []
            
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Mock existing laptop setup
            mock_laptop_setup = Mock()
            mock_laptop_setup.id = 1
            mock_db.query.return_value.filter.return_value.first.return_value = mock_laptop_setup
            
            result = await rag_engine.ingest_laptop_specs(
                brand="Dell",
                model_name="XPS 15 9520",
                user_id=1
            )
            
            assert result["message"] == "Laptop specs ingested successfully"
            assert result["laptop_setup_id"] == 1
    
    def test_extract_cpu_intel(self, rag_engine):
        """Test CPU extraction for Intel processors"""
        text = "Intel Core i7-12700H Processor, 14 cores"
        cpu = rag_engine._extract_cpu(text)
        assert cpu is not None
        assert "Intel" in cpu
        assert "i7" in cpu
    
    def test_extract_cpu_amd(self, rag_engine):
        """Test CPU extraction for AMD processors"""
        text = "AMD Ryzen 7 6800H processor"
        cpu = rag_engine._extract_cpu(text)
        assert cpu is not None
        assert "AMD" in cpu
        assert "Ryzen" in cpu
    
    def test_extract_cpu_apple(self, rag_engine):
        """Test CPU extraction for Apple Silicon"""
        text = "Apple M2 chip"
        cpu = rag_engine._extract_cpu(text)
        assert cpu is not None
        assert "Apple" in cpu
        assert "M2" in cpu
    
    def test_extract_gpu_nvidia(self, rag_engine):
        """Test GPU extraction for NVIDIA"""
        text = "NVIDIA GeForce RTX 3050 Ti"
        gpu = rag_engine._extract_gpu(text)
        assert gpu is not None
        assert "NVIDIA" in gpu
        assert "RTX" in gpu
    
    def test_extract_ram(self, rag_engine):
        """Test RAM extraction"""
        text = "16GB DDR5 RAM"
        ram = rag_engine._extract_ram(text)
        assert ram is not None
        assert "16GB" in ram
        assert "DDR5" in ram
    
    def test_extract_storage(self, rag_engine):
        """Test storage extraction"""
        text = "512GB NVMe SSD"
        storage = rag_engine._extract_storage(text)
        assert storage is not None
        assert "512GB" in storage
        assert "SSD" in storage
    
    def test_extract_display(self, rag_engine):
        """Test display extraction"""
        text = '15.6" FHD+ (1920x1200) IPS display'
        display = rag_engine._extract_display(text)
        assert display is not None
        assert "15.6" in display
    
    def test_extract_ports(self, rag_engine):
        """Test port extraction"""
        text = "USB-C Thunderbolt 4, USB-A 3.2, HDMI 2.0, SD card reader, 3.5mm audio jack"
        ports = rag_engine._extract_ports(text)
        assert len(ports) > 0
        assert any("USB-C" in p for p in ports)
        assert any("HDMI" in p for p in ports)
    
    def test_extract_os(self, rag_engine):
        """Test OS extraction"""
        text = "Windows 11 Pro"
        os = rag_engine._extract_os(text)
        assert os is not None
        assert "Windows" in os
    
    def test_extract_known_issues(self, rag_engine):
        """Test known issues extraction"""
        text = "Some users report battery drain issues and fan noise problems under load."
        issues = rag_engine._extract_known_issues(text)
        assert len(issues) > 0
        assert any("battery" in issue.lower() for issue in issues)
    
    def test_parse_specs_complete(self, rag_engine):
        """Test complete spec parsing"""
        scraped_data = [
            {
                'url': 'https://example.com/specs',
                'title': 'Test Laptop Specs',
                'content': 'Intel Core i7-12700H, 16GB DDR5 RAM, 512GB SSD, NVIDIA RTX 3050 Ti, 15.6" FHD display, Windows 11 Pro',
                'scraped_at': datetime.utcnow().isoformat()
            }
        ]
        
        specs = rag_engine._parse_specs(scraped_data, "Dell", "XPS 15")
        
        assert specs['brand'] == "Dell"
        assert specs['model'] == "XPS 15"
        assert specs['cpu'] is not None
        assert specs['ram'] is not None
        assert specs['storage'] is not None
        assert specs['gpu'] is not None
        assert specs['os'] is not None
    
    def test_store_in_vector_db(self, rag_engine):
        """Test storing specs in vector database"""
        specs = {
            'cpu': 'Intel Core i7-12700H',
            'gpu': 'NVIDIA RTX 3050 Ti',
            'ram': '16GB DDR5',
            'storage': '512GB SSD',
            'display': '15.6" FHD',
            'ports': ['USB-C', 'USB-A', 'HDMI'],
            'known_issues': ['Battery drain'],
            'source_urls': ['https://example.com/specs']
        }
        
        chunks_stored = rag_engine._store_in_vector_db(specs, "Dell", "XPS 15")
        
        assert chunks_stored > 0
        assert rag_engine.vector_store.add_documentation_chunk.called
        
        # Verify metadata
        call_args = rag_engine.vector_store.add_documentation_chunk.call_args
        metadata = call_args.kwargs['metadata']
        assert metadata['brand'] == "Dell"
        assert metadata['model'] == "XPS 15"
        assert 'section' in metadata
    
    def test_get_laptop_context_with_results(self, rag_engine):
        """Test retrieving laptop context from vector DB"""
        rag_engine.vector_store.get_chunks_by_laptop.return_value = [
            {
                'text': 'Intel Core i7-12700H processor',
                'metadata': {
                    'section': 'processor',
                    'source_urls': '["https://example.com/specs"]'
                },
                'score': 0.1
            }
        ]
        
        context = rag_engine.get_laptop_context("Dell", "XPS 15", "processor")
        
        assert "Dell XPS 15" in context
        assert "Intel Core i7" in context
        assert rag_engine.vector_store.get_chunks_by_laptop.called
    
    def test_get_laptop_context_no_results(self, rag_engine):
        """Test retrieving context when no data exists"""
        rag_engine.vector_store.get_chunks_by_laptop.return_value = []
        
        context = rag_engine.get_laptop_context("Unknown", "Brand", "query")
        
        assert "No specific documentation found" in context
    
    @pytest.mark.asyncio
    async def test_search_laptop_documentation(self, rag_engine):
        """Test web search for laptop documentation"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.text = """
            <html>
                <a class="result__a" href="https://www.dell.com/support">Dell XPS 15 Support</a>
                <a class="result__a" href="https://www.techradar.com/review">TechRadar Review</a>
            </html>
            """
            mock_get.return_value = mock_response
            
            results = await rag_engine._search_laptop_documentation("Dell", "XPS 15")
            
            assert len(results) > 0
            assert any('dell.com' in r['url'].lower() for r in results)
    
    @pytest.mark.asyncio
    async def test_scrape_specs_from_sources(self, rag_engine):
        """Test scraping specs from source URLs"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.text = """
            <html>
                <body>
                    <h1>Dell XPS 15 9520</h1>
                    <p>Intel Core i7-12700H, 16GB DDR5, 512GB SSD</p>
                </body>
            </html>
            """
            mock_get.return_value = mock_response
            
            sources = [
                {'url': 'https://www.dell.com/support', 'title': 'Dell Support'}
            ]
            
            scraped = await rag_engine._scrape_specs_from_sources(sources)
            
            assert len(scraped) > 0
            assert scraped[0]['url'] == 'https://www.dell.com/support'
            assert 'content' in scraped[0]


class TestVectorStore:
    """Test suite for ChromaDB Vector Store"""
    
    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create VectorStore with temporary directory"""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            store = VectorStore(persist_directory=str(tmp_path))
            store.collection = mock_collection
            return store
    
    def test_add_documentation_chunk(self, vector_store):
        """Test adding a documentation chunk"""
        result = vector_store.add_documentation_chunk(
            chunk_id="test_001",
            text="Intel Core i7 processor",
            metadata={'brand': 'Dell', 'model': 'XPS 15'}
        )
        
        assert result is True
        vector_store.collection.add.assert_called_once()
    
    def test_search_similar_chunks(self, vector_store):
        """Test searching for similar chunks"""
        vector_store.collection.query.return_value = {
            'documents': [['Intel Core i7 processor', '16GB RAM']],
            'metadatas': [[{'brand': 'Dell'}, {'brand': 'Dell'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = vector_store.search_similar_chunks("processor", n_results=2)
        
        assert len(results) == 2
        assert results[0]['text'] == 'Intel Core i7 processor'
        assert 'score' in results[0]
    
    def test_get_chunks_by_laptop(self, vector_store):
        """Test getting chunks by laptop model"""
        vector_store.collection.query.return_value = {
            'documents': [['Dell XPS 15 specs']],
            'metadatas': [[{'brand': 'Dell', 'model': 'XPS 15'}]],
            'distances': [[0.1]]
        }
        
        results = vector_store.get_chunks_by_laptop("Dell", "XPS 15", n_results=5)
        
        assert len(results) > 0
        vector_store.collection.query.assert_called_once()
    
    def test_get_stats(self, vector_store):
        """Test getting vector store statistics"""
        vector_store.collection.count.return_value = 42
        
        stats = vector_store.get_stats()
        
        assert stats['total_chunks'] == 42
        assert 'collection_name' in stats


class TestPhase1Integration:
    """Integration tests for Phase 1 requirements"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_ingestion(self):
        """
        Test complete ingestion flow:
        1. Ingest laptop specs
        2. Verify specs saved in MySQL
        3. Verify embeddings stored in ChromaDB with source URLs
        """
        with patch('backend.app.rag_engine.SessionLocal') as mock_db_class, \
             patch('backend.app.rag_engine.get_vector_store') as mock_vs:
            
            # Setup mocks
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            mock_vector_store = Mock()
            mock_vs.return_value = mock_vector_store
            
            # Mock laptop setup creation
            mock_laptop_setup = Mock()
            mock_laptop_setup.id = 1
            mock_laptop_setup.user_id = 1
            mock_laptop_setup.brand = "Dell"
            mock_laptop_setup.model = "XPS 15 9520"
            
            # Mock specs creation
            mock_specs = Mock()
            mock_specs.id = 1
            mock_specs.laptop_setup_id = 1
            mock_specs.cpu = "Intel Core i7-12700H"
            mock_specs.gpu = "NVIDIA RTX 3050 Ti"
            mock_specs.ram = "16GB DDR5"
            mock_specs.storage = "512GB SSD"
            mock_specs.source_urls = '["https://dell.com/specs"]'
            
            # Setup query side effects
            mock_db.query.return_value.filter.return_value.first.side_effect = [
                None,  # No existing laptop setup
                None,  # No existing specs
                mock_specs  # Return specs after creation
            ]
            
            # Create RAG engine and ingest
            from backend.app.rag_engine import RAGEngine
            engine = RAGEngine()
            engine.vector_store = mock_vector_store
            
            with patch.object(engine, '_search_laptop_documentation') as mock_search, \
                 patch.object(engine, '_scrape_specs_from_sources') as mock_scrape:
                
                mock_search.return_value = [
                    {'url': 'https://dell.com/specs', 'title': 'Dell Specs'}
                ]
                mock_scrape.return_value = [
                    {
                        'url': 'https://dell.com/specs',
                        'title': 'Dell Specs',
                        'content': 'Intel Core i7-12700H, 16GB DDR5 RAM, 512GB SSD, NVIDIA RTX 3050 Ti',
                        'scraped_at': datetime.utcnow().isoformat()
                    }
                ]
                
                result = await engine.ingest_laptop_specs("Dell", "XPS 15 9520", 1)
                
                # Verify MySQL storage
                assert mock_db.add.call_count >= 2  # LaptopSetup + LaptopSpecs
                assert mock_db.commit.called
                
                # Verify ChromaDB storage with source URLs
                assert mock_vector_store.add_documentation_chunk.called
                call_args = mock_vector_store.add_documentation_chunk.call_args
                metadata = call_args.kwargs['metadata']
                assert 'source_urls' in metadata
                assert metadata['source_urls'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
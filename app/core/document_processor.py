import tempfile
from pathlib import Path
from typing import BinaryIO
import shutil
from langchain_community.document_loaders import CSVLoader,PyPDFLoader,TextLoader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import get_logger

logger=get_logger(__name__)


class DocumentProcessor:
    SUPPORTED_EXTENSIONS={".pdf",".csv",".txt"}

    def __init__(self,
                 chunk_size: int | None = None,
                 chunk_overlap: int | None = None
                 ):

        settings=get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n","."," ",""],
            length_function=len,
        )

        logger.info(
            f"Document Processor initialized with chunk_size= {self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def load_pdf(self,file_path: str | Path) -> list[Document]:
        file_path=Path(file_path)

        logger.info(f"Loading PDF: {file_path.name}")

        loader=PyPDFLoader(str(file_path))
        documents=loader.load()

        logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
        return documents
    
    def load_texts(self,file_path: str | Path) -> list[Document]:
        file_path=Path(file_path)
        logger.info(f"Loading Text file: {file_path.name}")
        loader=TextLoader(str(file_path),encoding="utf-8")
        documents=loader.load()
        logger.info(f"Loaded Text file: {file_path.name}")
        return documents
    
    def load_csv(self,file_path: str | Path) -> list[Document]:
        file_path=Path(file_path)
        logger.info(f"Loading CSV: {file_path.name}")
        loader=CSVLoader(str(file_path),encoding="utf-8")
        documents=loader.load()
        logger.info(f"Loaded {len(documents)} rows from {file_path.name}")
        return documents
    
    def load_file(self,file_path: str | Path) ->list[Document]:
        file_path=Path(file_path)
        extension=file_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extension},"
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        
        loaders={
            ".pdf":self.load_pdf,
            ".txt":self.load_texts,
            ".csv":self.load_csv
        }

        documents = loaders[extension](file_path)
        for doc in documents:
            doc.metadata["source"]=file_path.name
            doc.metadata["file_type"]=extension

        return documents
    
    def load_from_upload(self,file: BinaryIO, file_name: str) -> list[Document]:

        extension= Path(file_name).suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported Extension: {extension}, "
                f"Supported Extensions: {self.SUPPORTED_EXTENSIONS}"
            )
        tmp_path=None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension
            ) as tmp_file:
                shutil.copyfileobj(file,tmp_file)
                tmp_path=tmp_file.name

            documents=self.load_file(tmp_path)
            for doc in documents:
                doc.metadata["source"] = file_name
                doc.metadata["file_type"] = extension
            
            return documents
        
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def split_documents(self,documents: list[Document]) -> list[Document]:
        logger.info(f"Splitting {len(documents)} documents into chunks")
        chunks=self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def process_file(self,file_path: str | Path) -> list[Document]:
        documents=self.load_file(file_path)
        return self.split_documents(documents)
    
    def process_upload(self,file: BinaryIO, file_name: str) -> list[Document]:
        documents=self.load_from_upload(file,file_name)
        return self.split_documents(documents)
    
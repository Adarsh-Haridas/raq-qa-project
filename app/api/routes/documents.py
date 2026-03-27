from fastapi import APIRouter,UploadFile,HTTPException,File

from app.api.schema import (
    DocumentUploadResponse,
    DocumentListingResponse,
    ErrorResponse,
)

from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStoreService
from app.config import get_settings
from app.utils.logger import get_logger

logger=get_logger(__name__)

router=APIRouter(prefix="/documents",tags=["Documents"])

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={
        400: {"model":ErrorResponse, "description":"Invalid file type"},
        500: {"model":ErrorResponse, "description":"Processing error"}
    },
    summary="Upload and ingest a document",
    description="Upload a document (PDF, TXT, or CSV) to be processed and added to the vector store.",
)

async def upload_file(file:UploadFile = File(description="file to upload")) -> DocumentUploadResponse:

    logger.info(f"Recieved document to upload: {file.filename}")

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )
    try:
        processor=DocumentProcessor()
        chunks=processor.process_upload(file.file,file.filename)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Chunks cannot be extracted from the file"
            )
        
        vector_store=VectorStoreService()
        document_ids=vector_store.add_documents(chunks)
        logger.info(
            f"Successfully processed: {file.filename}, "
            f"{len(chunks)} chunks & {len(document_ids)} documments"
        )

        return DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            chunks_created=len(chunks),
            document_ids=document_ids
        )
    
    except ValueError as e:
        logger.warning(f"Invalid file upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@router.get(
    "/info",
    response_model=DocumentListingResponse,
    summary="Get collection information",
    description="Get information about the document collection."
)

async def get_collection_info() -> DocumentListingResponse:

    logger.debug(f"Collection info requested")

    try:
        vector_store=VectorStoreService()
        info = vector_store.get_collection_info()

        return DocumentListingResponse(
            collection_name=info["name"],
            total_documents=info["points_count"],
            status=info["status"]
        )
    
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting collection info: {str(e)}"
        )
    

@router.delete(
    "/collection",
    responses={
        200: {"description":"Successfully deleted the collection"},
        500: {"model":ErrorResponse, "description":"Deletion Error"}
    }
)

async def delete_collection() ->dict:

    logger.warning("Collection deletion requested")

    try:
        vector_store=VectorStoreService()
        vector_store.delete_collection()

        return {"message":"Collection deleted succesfully"}
    
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting collection: {str(e)}"
        )
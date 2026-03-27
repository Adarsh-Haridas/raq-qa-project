import time
from fastapi import APIRouter,HTTPException
from fastapi.responses import StreamingResponse

from app.core.rag_chain import RAGChain
from app.api.schema import (
    QueryResponse,
    QueryRequest,
    SourceDocument,
    EvaluationScores,
    ErrorResponse
)
from app.utils.logger import get_logger
from app.config import get_settings

logger=get_logger(__name__)
router=APIRouter(prefix="/query",tags=["Query"])

@router.post(
    "",
    response_model=QueryResponse,
    responses={
        400: {"model":ErrorResponse, "description":"Inavlid request"},
        500: {"model":ErrorResponse, "description":"Query Processing error"}
    },
    summary="Ask a question",
    description="Submit a question and get an AI-generated answer based on the ingested documents."
)

async def query(request: QueryRequest) ->QueryResponse:

    logger.info(f"Quey received: {request.question[:150]}"
                f"(sources: {request.include_sources}, eval: {request.enable_evaluation})"
                )
    
    start_time=time.time()

    try:
        rag_chain=RAGChain()

        if request.enable_evaluation:
            result= await rag_chain.aquery_with_evaluation(
                request.question,
                request.include_sources
            )

            sources=(
                [
                    SourceDocument(
                        content=source["content"],
                        metadata=source["metadata"]
                    )
                    for source in result["sources"]
                ]
                if request.include_sources
                else None
                
            )

            answer=result["answer"]
            evaluation=EvaluationScores(**result['evaluation'])

        elif request.include_sources:
            result= await rag_chain.aquery_with_sources(request.question)
            sources=(
                [
                    SourceDocument(
                        content=source["content"],
                        metadata=source["metadata"]
                    )
                    for source in result["sources"]
                ]
                if request.include_sources
                else None
                
            )
            evaluation=None
            answer=result["answer"]

        else:
            answer= await rag_chain.aquery(request.question)
            sources=None
            evaluation=None
        
        processing_time_ms=(time.time()-start_time)*1000

        logger.info(f"Processed query in time: {processing_time_ms}"
                    f"(eval_included: {request.enable_evaluation})"
                    )
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources= sources,
            processing_time_ms=round(processing_time_ms,2),
            evaluation=evaluation
        )
    
        
    except Exception as e:
        logger.error(f"Error processing the query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the query: {str(e)}"
        )

@router.post(
    "/stream",
    responses={
        400: {"model":ErrorResponse, "description":"Invalid request"},
        500: {"model":ErrorResponse, "description": "Query processing error"}
    },
    summary="Ask a question (streaming)",
    description="Submit a question and get a streaming AI-generated answer."
)

async def query_stream(request: QueryRequest) ->StreamingResponse:

    logger.info(f"Streaming query received: {request.question[:150]}...")
    try:
        rag_chain=RAGChain()

        async def generate():
            try:
                async for chunk in rag_chain.chain.astream(request.question):
                    yield chunk
            except Exception as e:
                logger.error(f"Error in stream: {e}")
                yield f"\n\nError: {str(e)}"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain"
        )
    
    except Exception as e:
        logger.error(f"Error setting up stream: {e}")
        raise HTTPException(
            status_code=500,
            detail=f'Error processing query: {str(e)}'
        )
    

@router.post(
    "/search",
    responses={
        500: {"model":ErrorResponse, "description":"Search error"}
    },
    summary="Search documents",
    description="Search for relevant documents without generating an answer."
)

async def search_documents(request: QueryRequest) ->dict:
    logger.info(f"Search received: {request.question[:150]}...")

    try:
        from app.core.vector_store import VectorStoreService

        vector_store=VectorStoreService()
        results=vector_store.search_with_score(request.question)

        documents = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": round(score, 2)
            }
            for doc,score in results
        ]

        return {
            "query": request.question,
            "results": documents,
            "count": len(documents)
        }
    
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )
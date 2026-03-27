from datetime import datetime,timezone
from typing import Any

from pydantic import BaseModel,Field


#===========Health Schemas===========

class HealthResponse(BaseModel):

    status: str = Field(description="Service Status")
    timestamp: datetime = Field(default_factory= lambda: datetime.now(timezone.utc), description="Response Timestamp")
    version: str = Field(description="Application Version")

class ReadinessResponse(BaseModel):

    status: str = Field(description="Service Status")
    qdrant_connected: bool = Field(description="Qdrant Connection  Status")
    collection_info: dict = Field(description="Collection Information")


#===========Document Schemas===========

class DocumentUploadResponse(BaseModel):
    """Response after document upload"""

    message: str = Field(description="Status Message")
    filename: str = Field(description="Name of the File")
    chunks_created: int = Field(description="Number of chunks created")
    document_ids: list[str] = Field(description="List of document IDs")

class DocumentInfo(BaseModel):

    source: str = Field(description="Document Source/filename")
    metadata: dict[str,Any] = Field(default_factory=dict, description="Document Information")

class DocumentListingResponse(BaseModel):
    """Response for listing documents"""

    collection_name: str = Field(description="Collection name")
    total_documents: int = Field(description="Total documents count")
    status: str = Field(description="Collection Status")


#===========Query Schemas===========

class QueryRequest(BaseModel):
    """Request for RAG query"""

    question: str = Field(description="Question to ask",
                          min_length=1,
                          max_length=1000)
    include_sources: bool = Field(default=True,
                                 description="Include sources in RAG Response")
    enable_evaluation: bool = Field(default=False,
                                    description="Enable RAGAS Evaluation(faithfullness,answer relevance)")
    
    model_config={
        "json_schema_extra":{
            "examples":[
                {
                    "question":"What is RAG?",
                    "include_sources":True,
                    "enable_evaluation":False
                }
            ]
        }
    }

class SourceDocument(BaseModel):
    """Source Document Information"""

    content: str = Field(description="Document content excerpts")
    metadata : dict[str,Any] = Field(description="Document metadata")

class EvaluationScores(BaseModel):
    """RAGAS Evaluation Score"""

    faithfulness: float | None = Field(
        None,
        description="Faithfulness score (0-1): measures factual consistency with sources",
        ge=0.0,
        le=1.0,
    )
    answer_relevancy: float | None = Field(
        None,
        description="Answer relevancy score (0-1): measures relevance to question",
        ge=0.0,
        le=1.0
    )
    evaluation_time_ms: float | None = Field(
        None,
        description="Time taken for evaluation in milliseconds"
    )

    error : str | None = Field(
        None,
        description="Error message if evaluation failed"
    )

class QueryResponse(BaseModel):
    """Response for RAG query"""

    question: str = Field(description="Original question")
    answer: str = Field(description="Answer generated")
    sources: list[SourceDocument] | None = Field(
        None,
        description="Source documents used"
    )
    processing_time_ms: float | None = Field(
        None,
        description="Time for query processing in ms"
    )

    evaluation: EvaluationScores | None = Field(
        None,
        description="RAGAS Evaluation scores(if required)"
    ) 


#===========Error Schemas===========  

class ErrorResponse(BaseModel):
    """Error Response"""

    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    detail: str | None = Field(
        None,
        description="Detailed information of error"
    )

class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    
    error: str = Field(default="Validation Error" ,description="Error type")
    message: str = Field(description="Error message")
    errors: list[dict] = Field(description="Validation errors")
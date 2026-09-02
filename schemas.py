from pydantic import BaseModel, Field
from typing import List, Optional, Union

class AadhaarResult(BaseModel):
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    year_of_birth: Optional[str] = None
    gender: Optional[str] = None
    aadhaar_number: Optional[str] = None

class PANResult(BaseModel):
    name: Optional[str] = None
    father_name: Optional[str] = None
    pan_number: Optional[str] = None
    date_of_birth: Optional[str] = None

class DLResult(BaseModel):
    dl_number: Optional[str] = None
    name: Optional[str] = None
    dob: Optional[str] = None

class SubjectResult(BaseModel):
    subject: str
    marks: Optional[int] = None

class MarksheetResult(BaseModel):
    student_name: Optional[str] = None
    father_name: Optional[str] = None
    mother_name: Optional[str] = None
    college_name: Optional[str] = None
    subjects: List[SubjectResult] = []
    total_marks: Optional[int] = None
    sgpa: Optional[float] = None
    cgpa: Optional[float] = None
    university: Optional[str] = None
    board: Optional[str] = None
    semester: Optional[str] = None
    usn: Optional[str] = None
    register_number: Optional[str] = Field(None, alias="register number")

    class Config:
        populate_by_name = True

class OCRResponse(BaseModel):
    document_type: str
    extracted_fields: Union[AadhaarResult, PANResult, DLResult, MarksheetResult, dict] = Field(default_factory=dict)
    face_image: Optional[str] = None
    signature_image: Optional[str] = None
    job_id: str

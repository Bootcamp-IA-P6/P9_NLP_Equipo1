from pydantic import BaseModel, Field

# 1. Input schema: What the user sends to the API
class PredictionInput(BaseModel):
    text: str = Field(
        ..., 
        description="The raw comment or text to be analyzed",
        example="This is a test comment for analysis."
    )

# 2. Output schema: What our API returns structured
class PredictionOutput(BaseModel):
    is_toxic: bool = Field(
        ..., 
        description="Indicates if the text contains general toxic elements"
    )
    is_racist: bool = Field(
        ..., 
        description="Indicates if the text contains racist undertones"
    )
    is_hate_speech: bool = Field(
        ..., 
        description="Indicates if the text is classified as hate speech"
    )
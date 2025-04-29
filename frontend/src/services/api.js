// Service function to call your FastAPI
export async function analyzeStatement(formData) {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
      // Don't set Content-Type - browser will add the multipart boundary automatically
    });
  
    // Check if the response is successful
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
  
    // Parse and return the JSON response
    return await response.json();
  }
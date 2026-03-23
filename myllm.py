from openai import OpenAI
import os
import base64

class ChatGPTClient:
    def __init__(self, model="gpt-4.1-mini"):
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.model = model
        self.history = []
        
    

    def add_user_message(self, text):
        """Add a user message to history."""
        self.history.append({"role": "user", "content": text})


    def ask(self, prompt):
        """Send a prompt and return the assistant's reply."""
        # self.add_user_message(prompt)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=prompt
        )
        reply = completion.choices[0].message.content

        # Save assistant reply in history
        # self.history.append({"role": "assistant", "content": reply})

        return reply


    def ask_image_question(self, question, image_path):
        content = [{"type": "text", "text": question}]

        # Local file → convert to base64 with data URL format
        if image_path:
            with open(image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")
            
            # Determine image type from extension
            ext = image_path.lower().split('.')[-1]
            mime_type = f"image/{ext}" if ext in ['jpeg', 'jpg', 'png', 'gif', 'webp'] else "image/jpeg"
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_img}"
                }
            })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": content
            }]
        )

        return response.choices[0].message.content

    def ask_pdf_question(self, question, pdf_path):

        # 1. upload the pdf first
        uploaded_file = self.client.files.create(
            file=open(pdf_path, "rb"),
            purpose="assistants"
        )

        file_id = uploaded_file.id

        # 2. send chat request with file_id
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "file",
                            "file": {
                                "file_id": file_id
                            }
                        }
                    ]
                }
            ]
        )

        return response.choices[0].message.content


    def ask_text_question(self, text):
        content = [{"type": "text", "text": text}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": content
            }]
        )

        return response.choices[0].message.content

    def reset(self):
        """Clear the conversation."""
        self.history = []

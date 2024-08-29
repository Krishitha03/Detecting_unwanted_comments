from flask import Flask, render_template, render_template_string, request,send_file
import io
import plotly.express as px
import plotly.io as pio
import pandas as pd
import googleapiclient.discovery
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import TextVectorization # type: ignore

app = Flask(__name__)

# Load the pre-trained model
model = load_model("toxicity10epoch.h5")
sequence_length = 1800
MAX_FEATURES = 200000  # number of words in the vocab
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
df = pd.read_csv('train.csv')
X = df['comment_text']
y = df[df.columns[2:]]

vectorizer.adapt(X.values)

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyABP2SteqNGg1ucp3jk5EH9qtE7JqP_MKA"  # Replace with your actual API key

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/index')
def index():
    # Add your backend1 code here
    return render_template('index.html')

#getting input text and performing prediction
@app.route('/predict1', methods=['POST'])
def predict1():
    if request.method == 'POST':
        input_text = request.form['text']

        # Vectorize the input text
        input_text_vectorized = vectorizer([input_text])

        # Pad the sequence to match the expected sequence length
        input_text_padded = tf.keras.preprocessing.sequence.pad_sequences(
            input_text_vectorized.numpy(), maxlen=sequence_length
        )

        # Make predictions using the model
        res = model.predict(input_text_padded)

        # Convert predictions to binary (0 or 1) based on a threshold (0.5 in this case)
        binary_prediction = (res > 0.5).astype(int)

        # Create DataFrame
        output = pd.DataFrame({
            'Text': [input_text],
            'toxic': binary_prediction[0][0],
            'severe_toxic': binary_prediction[0][1],
            'obscene': binary_prediction[0][2],
            'threat': binary_prediction[0][3],
            'insult': binary_prediction[0][4],
            'identity_hate': binary_prediction[0][5]
        })

        return render_template('index.html', tables=[output.to_html(classes='data')], titles=output.columns.values)
    
@app.route('/YT')
def YT():
    # Add your backend1 code here
    return render_template('YT.html')

import re

def extract_video_id(youtube_url):
    # Regular expression to match the video ID
    video_id_match = re.search(r'(?<=v=)[^&#]+', youtube_url)
    
    if video_id_match:
        video_id = video_id_match.group(0)
        return video_id
    else:
        print("Invalid YouTube URL. Cannot extract video ID.")
        return None



@app.route('/toxic_comments', methods=['POST'])
def get_toxic_comments():
    url = request.form['videoId']
    videoId=extract_video_id(url)
    if videoId:
        print(f"Extracted video ID: {videoId}")

    video_request = youtube.commentThreads().list(
        part="snippet",
        videoId=videoId,
        maxResults=100
    )

    comments = []
    # Execute the request.
    response = video_request.execute()

    # Get the comments from the response.
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        public = item['snippet']['isPublic']
        comments.append([
            comment['authorDisplayName'],
            comment['publishedAt'],
            comment['likeCount'],
            comment['textOriginal'],
            public
        ])

    while 'nextPageToken' in response:
        nextPageToken = response['nextPageToken']
        nextRequest = youtube.commentThreads().list(
            part="snippet",
            videoId=videoId,
            maxResults=100,
            pageToken=nextPageToken
        )
        response = nextRequest.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            public = item['snippet']['isPublic']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['likeCount'],
                comment['textOriginal'],
                public
            ])

    # Convert comments to DataFrame
    dfr = pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'comments', 'public'])

    # Vectorize the comments
    input_text = vectorizer(np.array(dfr['comments']))  # Use the fitted vectorizer

    # Make predictions using the model
    res = model.predict(input_text)

    # Convert predictions to binary (0 or 1) based on a threshold (0.5 in this case)
    binary_predictions = (res > 0.5).astype(int)

    # Filter out toxic comments
    toxic_indices = np.where(np.any(binary_predictions == 1, axis=1))[0]
    toxic_comments_df = dfr.iloc[toxic_indices]
    # Filter out non-toxic comments
    nontoxic_indices = np.where(np.all(binary_predictions == 0, axis=1))[0]
    nontoxic_comments_df = dfr.iloc[nontoxic_indices]

    # Add prediction column to the DataFrame
    toxic_comments_df['prediction[toxic,severe_toxic,obscene,threat,insult,identity_hate]'] = binary_predictions[toxic_indices].tolist()

    toxic_count = len(toxic_comments_df)
    nontoxic_count = len(nontoxic_comments_df)

    # Create the pie chart
    labels = ['Toxic', 'Non-Toxic']
    values = [toxic_count, nontoxic_count]
    fig = px.pie(values=values, names=labels, title='Toxic vs Non-Toxic Comments')

    # Convert the plot to HTML
    pie_chart_html = pio.to_html(fig, full_html=False)

    # Convert DataFrame to HTML table
    toxic_comments_html = toxic_comments_df.to_html(border=2, index=False)

    full_html = f"""
    <html>
    <head>
        <title>Toxic Comments Analysis</title>
    </head>
    <body>
        <h1>Toxic vs Non-Toxic Comments</h1>
        {pie_chart_html}
        <h2>Toxic Comments</h2>
        {toxic_comments_html}
        <br>
        <a href="/download_toxic_comments" download><button>Download Toxic Comments as CSV</button></a>
    </body>
    </html>
    """
    # Store the DataFrame in session or a global variable
    global toxic_comments_df_global
    toxic_comments_df_global = toxic_comments_df  # Store for download
    return render_template_string(full_html) 

@app.route('/download_toxic_comments')
def download_toxic_comments():
    # Convert DataFrame to CSV
    csv_output = io.StringIO()
    toxic_comments_df_global.to_csv(csv_output, index=False)
    csv_output.seek(0)
    
    # Send the file to the user
    return send_file(
        io.BytesIO(csv_output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='toxic_comments.csv'
    )
if __name__ == '__main__':
    port=int(os.environ.get("PORT",4000))
    app.run(host='0.0.0.0',port=port)


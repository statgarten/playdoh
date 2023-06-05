import streamlit as st
import requests
import pandas as pd
import numpy as np
import altair as alt
import emoji

def main():
    st.header("Sentiment Analysis")
    left_column, right_column = st.columns(2)

    # Text input in the left column
    text = left_column.text_input('Enter your text here')

    # Button to start sentiment analysis
    if left_column.button('Analyze'):
        # Send POST request to the FastAPI server
        # response = requests.post('http://localhost:8001/sentiment_analysis', json={'text': text})

        # Get sentiments from the response
        # sentiments = response.json()['sentiments']

        # Create a DataFrame from the sentiments
        # df = pd.DataFrame(list(sentiments.items()), columns=['Sentiment', 'Value'])
        
        # df for UI test
        df  = pd.DataFrame({'sent': ['기쁨', '우울', '분노', '두려움', '사랑', '놀람', '중립'],
                            'emoji': ['☺️', '😟', '😡', '😱', '❤️', '😧', '😌'],
                            'prob': [0.064049, 0.045815, 0.585838, 0.000344, 0.000502, 0.301641, 0.001812],
                            })
        
        df = df.sort_values('emoji')
    
        colors = ["#FBCB0A", "#C70A80", "#3EC70B", "#590696", "#37E2D5", "#D3D3D3", "#FF4500"]
        color_scale = alt.Scale(domain=df['emoji'].unique(), range=colors)

        #  chart
        base = alt.Chart(df, height=550).mark_bar(size=30).encode(
                alt.X('prob', axis=None),
                alt.Y('emoji', title=''),
                alt.Color('emoji', scale=color_scale, legend=None)
                )

        # Add text to chart
        text = base.mark_text(dx=3, dy=-10, align='left', fontSize = 18).encode(
            text='prob'
        )

        chart = (base + text).configure_axis(
            grid=False
        ).configure_view(
            strokeWidth=0
        ).configure_axis(
            labelFontSize=30
            )

        right_column.altair_chart(chart, use_container_width=True)

# For running this file individually
# if __name__ == "__main__":
#     app()
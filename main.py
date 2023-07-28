from modeling import predict_input
import streamlit as st
from PIL import Image


##### CONFIG

# page config
st.set_page_config(page_title            = "Fake review prediction", 
                   page_icon             = ":pencil:", 
                   layout                = "centered", 
                   initial_sidebar_state = "collapsed", 
                   menu_items            = None)


# creating a function for prediction
def prediction(input_data):

    # loading the model
    model = predict_input.get_prediction(input_data)[0]

    if (model==0):
        return st.success('The review is Real')
    else:
        return st.warning('The review is Fake')


def main():

    # setting the title
    st.title('Fake Review Prediction')

    # image cover
    image = Image.open('image.jpg')
    st.image(image)

    # getting the input from the user
    Review = st.text_input('Product Review')

    # prediction code
    result = ''

    # creating a button for prediction
    if st.button('Submit'):
        return prediction(Review)


if __name__ == '__main__':
    st.main()



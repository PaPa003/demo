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
    model = predict_input.get_prediction(input_data)
    print(model)

    try:
      if (model[0]==0):
        return 'The Review is Real'

      else:
          return 'The Review is Fake'
      
    except TypeError as e:
      print(e)
      print("handled successfully")



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
        result = prediction(Review)

    st.success(result)




if __name__ == '__main__':
    main()



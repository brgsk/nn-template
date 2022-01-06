
# Streamlit
[Streamlit](https://docs.streamlit.io/) is an open-source Python library that makes
it easy to create and share beautiful, custom web apps for machine learning and data science.

In just a few minutes, you can build and deploy powerful data apps to:

- **Explore** your data
- **Interact** with your model
- **Analyze** your model behavior and input sensitivity
- **Showcase** your prototype with [awesome web apps](https://streamlit.io/gallery)

Moreover, Streamlit enables interactive development with automatic rerun on files changes.

Launch a minimal app with `PYTHONPATH=. streamlit run src/ui/run.py`. There is a built-in function to restore a model checkpoint stored on W&B, with automatic download if the checkpoint is not present in the local machine:

![](https://i.imgur.com/3lTnOA1.png)

import streamlit as st
import pands as pd

df = pd.read_cvs('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
df

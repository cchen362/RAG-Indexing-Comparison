#!/usr/bin/env python3
"""
Minimal dark mode test - completely isolated
"""
import streamlit as st
import time

st.set_page_config(page_title="Dark Mode Test", layout="wide")

# Force light mode with aggressive CSS
cache_buster = str(int(time.time()))

st.markdown(f"""
<style>
/* FORCE LIGHT MODE - NUCLEAR VERSION */
html, body, .stApp, [data-testid="stApp"] {{
    background-color: #ffffff !important;
    color: #000000 !important;
}}

/* All text - force black */
*, *::before, *::after {{
    color: #000000 !important;
    background-color: transparent !important;
}}

/* Headers */
h1, h2, h3 {{
    color: #0066cc !important;
}}

/* Cache buster: {cache_buster} */
</style>

<script>
console.log('Dark mode override loaded: {cache_buster}');
</script>
""", unsafe_allow_html=True)

st.title("🔍 Dark Mode Test")
st.write("**If you can read this text clearly in dark mode, the CSS is working.**")

st.subheader("Test Content")
st.write("This is regular text that should be black.")
st.write("This is another line of text.")

col1, col2 = st.columns(2)
with col1:
    st.write("**Column 1 text**")
    st.write("More text here")
    
with col2:
    st.write("**Column 2 text**") 
    st.write("Even more text")

# Test input
user_input = st.text_input("Test input (should have black text on white background)")
if user_input:
    st.write(f"You typed: {user_input}")

st.markdown("---")
st.markdown(f"""
**Debug Info:**
- Streamlit version: 1.48.1
- Cache buster: {cache_buster}
- CSS loaded at: {time.strftime('%H:%M:%S')}

**Instructions:**
1. Switch your browser to dark mode
2. If text appears white-on-light or is hard to read, the issue persists
3. If text is clearly visible, the CSS override is working
""")
mkdir -p ~/.streamlit/

echo "\
[theme]
primaryColor = '#667eea'
backgroundColor = '#ffffff'
secondaryBackgroundColor = '#f0f2f6'
textColor = '#0b1726'
font = 'sans serif'
" > ~/.streamlit/config.toml
```

### **Step 3: Update your project structure**

Your folder should look like this:
```
chatbot_project/
├── app.py                    (your streamlit app)
├── requirements.txt          (NEW - with torch)
├── setup.sh                  (NEW - for Streamlit Cloud)
├── vocab.pkl                 (your vocabulary file)
├── best_model.pt             (your model file)
└── .gitignore               (optional)
import pickle

# Open the .pkl file
with open('texts.pkl', 'rb') as file:
    data = pickle.load(file)

# Print the data
print(data)

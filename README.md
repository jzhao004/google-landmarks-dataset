# google-landmarks-dataset

Extracts information from Wikipedia for landmarks in [Google Landmarks v2](https://github.com/cvdfoundation/google-landmark)

- [train_clean.csv](https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv)
- [train_label_to_category.csv](https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv)

##### Files
scrape_wikipedia.py: Extracts the following information for each landmark
- Wikipedia page title
- First two sentences of Wikipedia summary
- Landmark coordinates (Latitude, longitude) if exists
- First two sentences of up to three Wikipedia sections

find_nearest_landmarks.py: Find top k nearest landmarks given coordinates

convert_to_json.py: Convert data to json format 

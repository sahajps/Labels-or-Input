echo "Running background description generator..."
python3 background_desc.py

echo "Running hate tagging..."
python3 hate_labelling.py

echo "Filtering data for meme augumentation..."
python3 data_for_regeneration.py

echo "Regeneration new non-hateful captions..."
python3 non_hateful_caption_generation.py

echo "Regenerating new meme with updated captions..."
python3 non_hateful_meme_generation.py
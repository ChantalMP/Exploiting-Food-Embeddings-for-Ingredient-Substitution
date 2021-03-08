# Exploiting Food Embeddings for Ingredient Substitution
Official repository of the paper "Exploiting Food Embeddings for Ingredient Substitution" ([Published at the International Conference on Health Informatics 2021](https://www.scitepress.org/Papers/2021/102020/102020.pdf)).

Identifying fitting substitutes for cooking ingredients can be beneficial for various goals, such as nutrient optimization, avoiding allergens, or adapting a recipe to personal preferences. In this repository, we present two models for ingredient embeddings, Food2Vec and FoodBERT. Additionally, we combine both approaches with images, resulting in two multimodal representation models. FoodBERT is furthermore used for relation extraction. According to a ground truth based evaluation and a human evaluation, FoodBERT, and especially its multimodal version, is best suited for substitute recommendations in dietary use cases.

### Installation:
1. Clone this repository
   ```
   git clone https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution
   ```
2. Install requirements:
   
   - Python 3.7 
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_lg
   ```
   
3. Download data and models
    
    - Download https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution/releases/download/0.1/food2vec_models.zip and place the content in ./food2vec/models
    - Download https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution/releases/download/0.1/foodbert_data.zip and place the content in ./foodbert/data
    - Download https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution/releases/download/0.1/foodbert_embeddings_data.zip and place the content in ./foodbert_embeddings/data
    - Download https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution/releases/download/0.1/multimodal_data.zip and place the content in ./multimodal/data
    - Download https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution/releases/download/0.1/relation_extraction_models.zip and place the content in ./relation_extraction/models
        
4. Optional: Generate data for FoodBERT and RE training
    - First, get the [Recipe1M+ dataset by Marin et al.](http://pic2recipe.csail.mit.edu) from http://im2recipe.csail.mit.edu/dataset/login/ (login required; as of November 2020, the correct link is: Layers (381 MiB))
    - Unzip, rename layer1.json to recipe1m.json and place it in .data/
    - Afterwards run 
    ```
    python -m normalisation.normalize_recipe_instructions
    python -m foodbert.preprocess_instructions
    ```
   
   Only for RE training:
   - Sadly, we can not publish the comment data needed for the relation extraction model
   - If you want to train or use the relation extraction model to generate substitutes, you need to scrape comments yourself. The scripts for this are provided as is, but they are not maintained.
   - All scripts can be found in comment_scraping.
   
5. Evaluation:
   - We can't make our ground-truth public, but if you want to reproduce our results or compare your own method, it is available upon request.

### Usage
1. Human and ground-truth-based evaluation: see [evaluation/README.md](evaluation/README.md)
2. Food2Vec training and substitute generation: see [food2vec/README.md](food2vec/README.md)
3. FoodBERT training: see [foodbert/README.md](foodbert/README.md)
4. FoodBERT substitute generation: see see [foodbert_embeddings/README.md](foodbert_embeddings/README.md)
5. Generating image embeddings for multimodal approaches: see see [multimodal/README.md](multimodal/README.md)
6. Data normalisation: see [normalisation/README.md](normalisation/README.md)
7. Relation Extraction training and substitute generation: see [relation_extraction/README.md](relation_extraction/README.md)

### Colab Examples
Using FoodBERT: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dnf8Wl2A_Mf_tUd6OUhNayuK_tfBm12w?usp=sharing)  
Using Food2Vec: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZSx5-_awCiccVlmhjnWZPA7qS4LHCWTY?usp=sharing)  
Using Image Embeddings: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dVIt15PyMWBl9u7m93Y5SgMS_ThhYX7I?usp=sharing)  
Generate Substitutes - FoodBERT: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tC7DRHUQx4qCrOAWhqdW1uuIwT1jyc1f?usp=sharing)  
Generate Substitutes - Food2Vec: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vv2zTAuMD1SpNiUhp-J-NDwbVx7Fjtoe?usp=sharing)  

If you encounter any problems with the code, feel free to contact us at {chantal.pellegrini, ege.oezsoy, monika.wintergerst}[at]tum[dot]de.

<!---
add citation when published
-->


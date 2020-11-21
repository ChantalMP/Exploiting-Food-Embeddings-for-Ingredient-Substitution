## Summary
Generates image-based embeddings for ingredients.

## How to Run
We can not provide the ingredient images used for generation due to copyright. You can download your own images for every ingredient and put them inside a folder named downloads/{$Ingredient_name} which should be in /data. However, we recommend to directly use the provided embedding_dict.pth

Then you can run
    
    python -m multimodal.imagenet_embeddings
 
The resulting dict will be saved in multimodal/data.

The rest of the approaches directly work with this dict.

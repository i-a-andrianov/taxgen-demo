import os
os.environ["CUDA_VISIBLE_DEVICES"] ="3"

from diffusers import DiffusionPipeline
from nltk.corpus import wordnet as wn
from tqdm import tqdm


pipe = DiffusionPipeline.from_pretrained("playgroundai/playground-v2-1024px-aesthetic").to('cuda')

prompts = []
filenames = []

def traverse_taxonomy(start=wn.synset('entity.n.01')):
    prompt = "An image of " + start.lemmas()[0].name().replace('_', ' ') + " (" + start.definition() + ")"
    filename = os.path.join("./images", "n{:08d}_generated.JPEG".format(start.offset()))

    if not os.path.exists(os.path.join("images", "n{:08d}.JPEG".format(start.offset()))):
        if not os.path.exists(os.path.join("images", "n{:08d}_generated.JPEG".format(start.offset()))):
            prompts.append(prompt)
            filenames.append(filename)
    
    for hyponym in start.hyponyms():
        traverse_taxonomy(hyponym) 

traverse_taxonomy()
print(len(prompts), len(filenames))

batch = 4
for i in tqdm(range(0, len(prompts), batch)):
    images = pipe(prompts[i:i+batch], num_inference_steps=10,
                                guidance_scale=7.0).images
    for image, filename in zip(images, filenames[i:i+batch]):
        image.save(filename, format='JPEG')

import os


if os.path.exists(f"./resume_iterator.json"):
    os.remove(f"./resume_iterator.json")
if os.path.exists(f"./postsType.json"):
    os.remove(f"./postsType.json")
if os.path.exists(f"./posts.json"):
    os.remove(f"./posts.json")
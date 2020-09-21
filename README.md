# Developing a hybrid recommender system for Amazon products (Sports & Outdoors)

## Introduction:

In recent decades, many web services such as Netflix, Youtube, Amazon, and other companies developed/implemented systems to suggest relevant products to their users. Content and collaborative recommenders are among the most popular methods for many web services.

These recommenders, however, have their drawbacks. For example, the collaborative filters are not working well to suggest new items to users. Though the context filters can better handle the cold start problem, these filters can only recommend an item whenever there are other items with features similar to the new one. To address the mentioned limitations, we can use hybrid recommenders. As such, for this project, I aimed to build a hybrid recommendation system.

## Methods:

To achieve the main goal of this project, I used a Amazon review dataset (https://nijianmo.github.io/amazon/index.html). This dataset includes reviews (ratings, text, helpfulness votes) of different product categories such as Amazon fashion, beauty, books, electronics, etc. Here, I mainly focused on Sport & outdoors category. This dataset includes reviews (ratings, text, helpfulness votes), product metadata (e.g., product information: color , size (large or small), package type; and Product images that are taken after the user received the product), and links.

As mentioned above, to develop a recommender system, I combined two systems: collaborative and recommender systems. In the content system, I will use an autoencoder to perform the compression between different reviews. I will also use a neural network to find products and user embedding. To combine these two systems, I will average cosine similarities.
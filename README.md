# Developing a hybrid recommender system for Amazon products: Sports & Outdoors



In recent decades, many web services such as Netflix, Youtube, Amazon, and other companies developed/implemented systems to suggest relevant products to their users. Content and collaborative recommenders are among the most popular methods for many web services.

These recommenders, however, have their drawbacks. For example, the collaborative filters are not working well to suggest new items to users. Though the context filters can better handle the cold start problem, these filters can only recommend an item whenever there are other items with features similar to the new one. To address the mentioned limitations, we can use hybrid recommenders. As such, for this project, I aimed to build a hybrid recommendation system.


To achieve the main goal of this project, I used an Amazon review dataset (https://nijianmo.github.io/amazon/index.html). This dataset includes reviews (ratings, text, helpfulness votes) of different product categories such as Amazon fashion, beauty, books, electronics, etc. Here, I mainly focused on Sport & outdoors category. This dataset includes reviews (ratings, text, helpfulness votes), product metadata (e.g., product information: color , size (large or small), package type; and Product images that are taken after the user received the product), and links.

As mentioned above, to develop a recommender system, I combined two systems: collaborative and recommender systems. In the content system, I used embedding neural networks to extract features of the review text. I also vectorized items text by using tfidf method. Finally, I combined these features and measured cosin similarity matrix to find similar items.

For the collaborative system, I used the non-negative matrix factorization algorithm to extract latent features and used these features to find top items for a given user based on his/her previous rating. In the hybrid algorithm, the collaborative filter extracts a top item that a user may like. Then, the content-based system finds similar items to the item that the collaborative suggested. 

I deployed the hybrid system in an AWS ec2 instance ( see [&lt;here&gt;](http://3.84.68.105:8080/) ). For furthure information, please refer to the related slides ([&lt;here&gt;](Slides/Amazon_recommenders.pdf))

<p align="center">
<img src='Figs/website.png'>
<center>Fig 1. A screenshot of the deployed product. </em>
</p>


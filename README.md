# Bookwyrm
Bookwyrm is a project developed in the context of the Fuzzy Sets class. The goal is to make a small app or plugin to help people find new fantasy book suggestions.

## 0. How to use the programm

**Standard User:**<br>

1. Run Bookwyrm.ipynb, the requiered libraries can be found in requirments.txt
2. Insert your custom values or select one of the books at the top to get recommendation. If the "Specify number of clusters" is left at 0 the programm will search for a good amount of clusters to generate, but from experience using 15-20 clusters leads to more interesting and varied results.
3. Click on "Get Recommendations". The programm will compute 4 different lists, one list only taking the moods into acount, one list for moods+rules, one list for moods+rating and one list for moods+rules+rating. You can change the inputs at any time and click on "Get Recommendation" again to refresh the book suggestions.

**For testing:**<br>
<br>
In the bookwyrm_analysis.ipynb you can see how we tested the programm, including how the clusters are generated.

## 1. Problem Identification and Motivation

The initial idea was to make an app to suggest fantasy books to a reader. We thought of using the book libraries of the user and/or a questionnaire as input to our app to generate suggestions. Interviews we made provided us with category ideas to focus on for helping in the selection.

This led to the creation of a simple prototype which was shown in the presentation and afterwards to some friends. The feedback from these events made us recognise that our initial idea was lacking a special selling point. The idea we came up with is to focus primarily on the moods the books convey. So instead of categorizing a book via rating and reviews we mainly categorize the moods of the book. This was made possible by finding an interesting site [The StoryGraph](https://app.thestorygraph.com/) where readers rate the books by their moods.


## 2. Define the objectives for a solution

With this new idea we created a new prototype. The prototype is going to make use of fuzzy clustering to compare the moods ratings of the books with either a questionnaire or the users library. 

For the questionnaire we have two ideas at this moment, which will have to be evaluated.<br>
- The users will be asked for each mood/pace how they feel about it and with that result we are going to calculate a vector to compare with our library.<br>
- The users will get a more fuzzy questionnaire where each question can give rating to more then one mood. The results of the questionnaire will then be aggregated to calculate a vector to compare with our library. This was scrapped during the implemtation due to being to complicated and weird to use.

## 3. Design and development

### Data

The dataset used in this project comes from [The StoryGraph](https://app.thestorygraph.com/). This website provides detailed information and analytics about books. It aggregates user-generated impressions and reading experiences. In particular, it provides statistics on how readers felt about each book, including:
- **Moods** (reflective, emotional, adventurous,...)
- **Pace** (slow, medium, fast)
- **Additional attributes** such as genre tags, content warnings, length and community ratings.
  
These features make The StoryGraph the perfect resource for building recommendation or classification systems.

The data is stored in a CSV file with currently the following categories:

**Main:**
- Title
- Author
- Length

**Moods:**
- Adventurous
- Challenging
- Dark
- Emotional
- Funny
- Hopeful
- Informative
- Inspiring
- Lighthearted
- Mysterious
- Reflective
- Relaxing
- Sad
- Tense

**Pace:**
- Slow
- Medium
- Fast

We gathered the data of 100 books from the website. If the book is part of a series we only ever took the first book of that series. You can check-out the data in book_list.csv.

### Algorithm basic setup

An interactive jupyter notebook was created to test our artefact.

Setup:<br>
1. We create fuzzy clusters with the moods from the books from the database. The algorithm uses an fpc test to determine the optimal amount of clusters (user testing showed that while this is theoretically optimal, for the user experience more clusters seem more interesting).
2. The mood input from the user are then fitted into the cluster to determine from which fuzzy cluster the suggestions should be generated. This return a list of all the books with an assigned cluster_strenght to how well they fit with the user input vector.
2. For the recomendation with rules we use fuzzy rules to give the books a scoring depending on the prefered lenght and pace inputs from the user. This score then gets mutiplied with the cluster_strenght to determine a ranking. 
3. For the recomendation with rating we simply multiply the cluster_strenght of each book with their rating divided by 5.


## 4. Demonstration

We conducted several interviews with different people, mostly people from which we knew do read fantasy books (7 interviews 14.12.2025). Those interviews can be found under INTERVIEW.md.



# Litterature
- Brocke, Jan vom & Hevner, Alan & Maedche, Alexander. (2020). Introduction to Design Science Research. 10.1007/978-3-030-46781-4_1.
- The StoryGraph website [https://app.thestorygraph.com/](https://app.thestorygraph.com/), © 2025 The StoryGraph Ltd.

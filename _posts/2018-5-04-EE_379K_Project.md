---
layout: post
title:  "I Only Love My LSTMs And My N-Grams - Drake Lyrics Generator"
date:   2018-05-04 01:43:30 -0500
categories: Data Science
---
##### _A machine learning project to produce Drake’s lyrics and possibly write Take Care Part 2_

![Drake]({{ "/assets/drake.jpg" | absolute_url }})
<hr><br>
Team Members:
Julian Domingo, 
Karthik Konath, 
Alex Ma, 
Kyle Polansky

Github Link: <https://github.com/JulianDomingo/drake_lyrics> 

_If you would like to view our presentation for the project, [click here!](https://docs.google.com/presentation/d/1U_RLGtldaigLLoWls3lVqpbYNPO7oGhA-O6Njz2CeEA/edit?usp=sharing)_
<hr><br>

# Motivation
For our Data Science Laboratory class, EE 379K, we were tasked with a project to demonstrate the machine learning techniques that we have learned from the course throughout the semester. After exploring the plethora of options for our assignment, we decided to create a rap lyric generator that closely mimics the popular rap artist Drake.

One interesting idea we had in mind was to venture into ghostwriting, which is an interesting topic that consistently come up in rap culture. Many people believe that ghostwriting isn't being a true and authentic artist if one isn't writing their own lyrics, but many successful rappers have had reached top charts with songs that they never even wrote. 

Drake has been part of the ghostwriting scene, being known for his [verbal battles with Meek Mill.](https://www.washingtonpost.com/news/morning-mix/wp/2016/05/26/drake-meek-mill-and-their-bitter-feud-about-ghostwriting-in-hip-hop/?noredirect=on&utm_term=.26a588b5266e)

![Meek Mill's tweet]({{ "/assets/meekmill.png" | absolute_url }})

For better or worse, ghostwriters are secretly being used throughout all genres of the music industry. We thought it would be cool to possibly make our own A.I. ghostwriter to recreate lyrics that could be the next club banger, as well as further understand the methods of Natural Language Processing/Generation. We will also dive deeper into machine learning frameworks such as PyTorch and Tensorlow.

# Data Collection
Our project requires lyrics from songs written by Drake. In addition to raw lyrics, we save verse markers to help identify song composition. Finally we save standard metadata like the song name and album that it originated in.

We started by using a simple Spotify API search `q="artist:Drake"` to list all of Drake's songs. We removed some of the extraneous punctuation to standardize the titles.

With a list of Drake's songs, we used Beautiful Soup to scrape lyrics off MetroLryics and Genius when the lyrics couldn't be found on MetroLryics. Our scraper is based off a [sample](https://towardsdatascience.com/generating-drake-rap-lyrics-using-language-models-and-lstms-8725d71b1b12 "sample") by Ruslan Nikolaev, but expanded to support more lyrics websites and a more dynamic song list. We ended up with 139 songs using this method.

In the future, more advanced song searches could be performed for additional lyric data, such as looking for unreleased songs or ones where Drake is a featured artist. If this approch is pursued, it's important to be careful only to select specific verses that Drake wrote and preformed.


# Preprocessing

![Data]({{ "/assets/data.png" | absolute_url }})
| *Excuse the language* |

With our dataset aquired, we ran it through a few preprocessing steps. The first is to replace all newline characters with a more visible `|-|` character that denotes the end of a line. This character is treated more like a word than a whitespace character as it greatly effects the song rhythm and flow.

Next up we normalized words in the lyrics. For example, the words yeah, yuh, and yea are all lexically identical. As our models focused primarily on character and word structure, we also removed most punctuation such as trailing commas and question marks. We also made all words lowercase in order to maintain uniformity when the lyrics are generated. 

Finally, we noticed that a few songs were released both as a single and part of an album, and hence in our data set twice. We manually removed these duplicate songs so they don't adversely effect the algorithm.


# Now Let's Get to the Machine Learning!


# N-Gram Markov Chain Models

#### Overview
The first approach we used to generate Drake's lyrics were N-Gram Markov Chains. We converted the pre-processed lyrics into order 1,2, and 3 n-grams so we would test how varying the order of the N-Gram Markov chain changes the predicted sentence.![N-Grams](https://i.stack.imgur.com/8ARA1.png)
We then generated Markov Chains with the probability of changing to a particular state being proportional to the number of times the next state succeeded the previous state in the corpus. We continue advancing states until an end of verse marker is found in the next state.

![N-gram Markov Chain](https://qph.fs.quoracdn.net/main-qimg-9b40cfdcc84e92063a499fbd88da3222)
#### Results

###### n = 1

> **Shawty Wanna come through  
and I go another  
And Tell me  
She gon make it time, yeah, yeah, yeah  
Who you know who"  
I got his music, when you n\*\*\*\*\*  
Trust this s\*\*t I heard that I'm on  
So yeah, yeah  
That's the glo'ray  
You would shut it for the same attention  
I gave up that's it now?**

You can rename the current file by clicking the file name in the navigation bar or by clicking the **Rename** button in the file explorer.

###### n = 2

> **Go and own that s\*\*t the calm  
Dedicated to my self u deceiving  
You spent your last fifty (ugh, cash)  
I f\*\*k the shade they throwin'  
Understand girl, we fam' girl  
Tell your big homie I'm all the money in the way  
Real n\*\*\*a what's up? what's up? what's up?  
Now it's "F**k you, I wouldn't doubt it cause all y'all n\*\*\*\*s  
All the girls they got it  
Wayne put me right  
F\*\*kin' with the kid and pray for your n\*\*\*a  
F\*\*kin' somethin', yeah n\*\*\*a, f\*\*kin' somethin'  
If there's hell below. I'll see you doin' well, doin' well dog  
Blew up and pray for the city, I let that slip  
Deceiving you are**

###### n = 3

> **Yeah, Lickwood means rewind, a gunshot means forward  
You wanna do is always on, man, I guess that makes u Aunt Viv  
Until u take from me  
Ever since you picked up, lucky that you love me enough for two n\*\*\*\*s boy stunt double  
Fame is like a video  
Bunch of underground kings, thought you wanted yours like I should dust something  
What am I in class if this is staged  
I'm on fire  
Tell me how to say no  
Are we still good? Are we still breathin'  
And its all because you chose a side  
Kick game, run it real from the basement  
That's why you gotta go twice  
28 at midnight, wonder what's next for me  
Cause to her and told me she prayed it  
Can't have everything  
You know it, oh ho, you know it, f\*\*k them stories, f\*\*k the f\*\*k-s\*\*t, I'm biased  
I miss you  
That's that s\*\*t**

#### Order Comparison

It seemed order 2 N-gram Markov Chains performed the best in terms of balancing creativity vs coherence and similarity to Drake's own lyrics. When we moved to order 3 chains, lyrics were essentially just being copied from songs completely, while order 1 chains lacked the coherence and similarity that we desired. However, considering that every word choice was randomly based off only the previous one for order 1 chains, the results were surprisingly good.



# LSTM RNNs (Long Short-Term Memory Recurrent Neural Nets)

#### Overview

The other approach we used to generate lyrics was through a RNN, but specifically an **LSTM** RNN. The primary difference in the LSTM is its inherent ability to learn long-term dependencies in the data, whereas a regular RNN is severely limited in this regard.

This prominent difference is due to the increased complexity of the network's update equation:

![Regular RNN]({{ "/assets/regular_rnn.png" | absolute_url }})

As seen above, a regular RNN simply contains repeating, single **tanh** layers.

![LSTM RNN]({{ "/assets/lstm_rnn.png" | absolute_url }})

On the other hand, LSTM layers contain four, interacting neural network layers of point-wise operations (i.e. vector addition), concatenation, and copying of vectors to different locations within each network cell.

#### Character-Level vs. Word-Level Models

We trained two types of LSTM RNNs: one models the data at a *character* level, predicting the subsequent **character** given a pre-specified input sequence; the other models the data at a *word* level, predicting the subsequent **word** given a pre-specified input sequence. This prediction is repeatedly performed until the specified character or word length threshold is reached.

#### Softmax Classifier (Cross-Entropy Loss)

We run the same training process for both the character-level and word-level RNN: given an input sequence (i.e. "The sky is"), we one-hot encode each character (or word) into individual vectors, then feed each into the RNN iteratively. This ultimately produces a sequence of **n**-dimensional output vectors (one for each character or word in the input sequence), containing confidence scores the RNN assigns for the subsequent character or word in the sequence:

![Training Process]({{ "/assets/training.jpeg" | absolute_url }})

This confidence score assignment is repeated continuously until the scores converge, determined through the LSTM's backpropagation algorithm which determines what directions the LSTM should adjust the network weights to. Ultimately, this increases the confidence of the "expected" subsequent character or word. In our project's case, we use **mini-batch gradient descent** to accomplish this goal. Compared to stochastic gradient descent and batch gradient descent, mini-batch succeeds in its ability to more quickly converge the confidence scores, while its increased frequency in updating the model's states allows computational efficiency due to not having all training data in memory. This helped tremendously in improving our model's predictive capabilities, especially due to our time constraints (Tuesday team rip).

#### Perplexity Scoring

The metric we use to determine the "correctness" of our model's predictions is through the **perplexity** function, shown below:

![Perplexity]({{ "/assets/perplexity.png" | absolute_url }})

where ***p<sub>target<sub>i</sub></sub>*** is the probability of character or word ***i*** occurring as the prediction. The character-level RNN converged to a lower perplexity score in comparison to the word-level RNN, shown below:

![Training Loss - Perplexity]({{ "/assets/training_perplexity.png" | absolute_url }})

#### Results!

#### Word-Level Model

> Canada, coming Griffith baritone, friend's<br>
> Gotta roll a Wraith n\*\*\*\*\* to aim for soon on y'all back after now<br>
> Now you're Toaster, and overlook my Girls from Care<br>
> I'm a good girl too got you had the dead<br>
> Mothaf\*\*\*a feel let 'em knew good some same girls, ever make it like

As seen above, the word-level model doesn't capture grammatical syntax very well. We concluded it is primarily due to the size of our training data, as Drake only has ~140 songs released by himself + singles.

#### Character-Level Model

###### Epoch 1
> Why it wacting you frop tels and bezy so<br>
> That’s you prised<br>
> Intuigh ofrest<br>
> And o lith be’ve the lasle

###### Epoch 25
> Yeah dink ya is laying all you hands<br>
> Why I’m in the virtion that right?<br>
> When she could ever put the costs with me<br>
> I’m the one she can so getting in my knees<br>
> When I be right around my hands for my own peace”

###### Epoch 50
> And I’m feeling so low with a good go<br>
> That’s why I ain’t all that I made<br>
> I just need a way, on and on I about to lose everyday

For the character-level model, the RNN progressively begins to understand grammatical syntax, and at epoch **50**, it begins to model internal rhyming. We concluded that the character-level RNN ended up performing significantly better after a long period of training due to the smaller size of our training corpus - with more data to train on, the word-level model would likely perform at a similar level as the character-level RNN.



# Fun Facts
Our dataset contains 75,121 words, with only 5908 unique words. The top 10 most common words are:

| Word  | Count  |
| ------------ | ------------ |
| i  | 3266  |
| you  | 2591  |
| the  | 2402  |
| to  | 1480  |
| and  | 1440  |
| it  | 1356  |
| a  | 1341  |
| me  | 1319  |
| im  | 1198  |
| my  | 1051  |

It's interesting that many of the most commonly used words are quite short. In fact, nearly 73% of our dataset is composed of words that are 4 characters or less. We can hypothesize that Drake's structure is very simple and contains many link words.

Moving onto N-Grams, here are the top 10 most common bigrams in our dataset:

| Bigram  | Count  |
| ------------ | ------------ |
| 'you', 'know'  | 224  |
| 'in', 'the'  | 222  |
| 'i', 'got'  | 213  |
| 'i', 'know'  | 188  |
| 'i', 'dont'  | 172  |
| 'and', 'i'  | 164  |
| 'i', 'just'  | 163  |
| 'on', 'the'  | 153  |
| 'it', 'i'  | 127  |
| 'im', 'on'  | 122  |


And the top 10 most common trigrams:

| Trigram  | Count  |
| ------------ | ------------ |
| 'versace', 'versace', 'versace'  | 72  |
| 'yeah', 'yeah', 'yeah'  | 70  |
| 'i', 'get', 'it'  | 60  |
| 'woah', 'woah', 'woah'  | 53  |
| 'its', 'okay', 'its'  | 50  |
| 'okay', 'its', 'okay'  | 50  |
| 'im', 'on', 'im'  | 50  |
| 'on', 'im', 'on'  | 50  |
| 'get', 'it', 'i'  | 46  |
| 'own', 'it', 'own'  | 46  |

Looking at the most common bigrams, we see that Drake commonly talks about himself and someone else "you" quite frequently in his music. The trigrams don't reveal a whole lot about the dataset in general, but rather expose patterns that are very common in his songs. For example, the song Versace has the trigram 'versace', 'versace', 'versace' 72 times in the song. 


# Conclusion

To conclude, we created a rap lyric generator that closely mimics rap artist Drake by using a LSTM (Long Short-Term Memory) neural net by scraping for his lyrics from sites such as Genius and MetroLyrics. We also explored the options of using Markov Chains to generate lyrics that could possibly be more coherent.

In the future, we hope to look deeper into implementing a rhyming scheme, and to have the model generate verses and hooks to create its own rap song. In general, this model and technique could be applied to different music artists and could be expanded to create lyrics certain genres. We found that this [Stanford research paper](https://nlp.stanford.edu/courses/cs224n/2009/fp/5.pdf) could help us achieve that goal.

It's also important to mention that the dataset for Drake's lyrics is still a bit too small for our liking. Using an artist that has significantly more lyrics and word content would potentially create a robust lyrics. Another alternative would be to take lyrics from multiple rap artists, or the as much hip hop artists as possible.

In all, we hope that some day an artist will utilize machine learning and neural networks to potentially create lyrics that could one day hit #1 on Billboard's Hot 100.


_Thanks for reading! :)_

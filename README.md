Git repo for Math 189 midterm project. Goal: use machine learning to
try and predict stock prices. General architecture: will create some
form of network to analyze the stock data itself, and another to parse
webscraped news articles from financial websites (and, possibly,
academic journals).

# General game plan
1. Web scrape financial data from some non-empty subset of these
   websites:
   i. morningstar
   ii. Seeking alpha
   iii. Yahoo finance
   iv. Google stocks

2. Sample price data from various-scale time windows, weighting more
   heavily towards the short-term. E.g., 1000 points from today, 500
   (moving averaged) from over the last week, 500 uniformly sampled
   over the past 30 days.

   Each point in these vectors will be a vector structured similarly
   to `(price per share, avg. time to completion, total shares sold)`,
   or something. We'll figure out the details later. These will be fed
   into some sort of machine learning algorithm. We might also include
   some information from an ETF/ETP representative of the industry
   this stock was from.

   Furthermore, we might try and scrape financial news articles about
   the industry / stock we're looking at, and apply some natural
   language processing libraries to get word vectors of adjectives and
   assign them numerical values of "favorable" or not. This will
   result in a reaaaally lossy "opinion" number that we can append to
   the data being fed into the model, etc.

3. Given all this input information, the model will attempt to do one
   of the following:

   i. (prototype goal) predict whether the stock will be up or down in
   10 minutes

   ii. (more long-term project goal) find a basic trading strategy.
   I.e., perform one of the following:
       a. Choose to buy (long)
       b. Choose to buy (short)
       c. Choose to sell
       d. Choose to do nothing
   Cost function will be determined based on whether the model "made" or
   lost money, and possibly the order of magnitude of the profit /
   loss.

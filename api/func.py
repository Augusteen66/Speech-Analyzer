################### This script will contain the utility functions ################################

## 1. Text Summarizer ##
from gensim.summarization.summarizer import summarize

def summ(doc):
    return summarize(doc)

## 2. Text2Emotion
import text2emotion as te

def text2emo(doc):
    return te.get_emotion(doc)

## 3. Sentiment Intensity Analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

def sia(doc):
    return vader.polarity_scores(doc)


doc='''David Perdue: (00:01)
How great is it to be back in Macon, Georgia with Donald Trump coming to Macon again? I want to thank Governor Kemp because he laid it out just now, but let me remind you of something that happened in 2002. Sunny just mentioned it. Because of you, we turned this thing around, and for the last 18 years, we have built a state that’s now the number one state in the country in which to do business.
David Perdue: (00:25)
That did not happen by accident. Brian Kemp is the third governor in a row that understands how to do that. And thanks to you, that’s going to continue. Today, we have an opportunity. Every generation has an opportunity like this. They have their moment of crisis. With Donald Trump, it’s all about promises made and promises delivered. After eight years of Joe Biden, you hear Joe lately? Joe said the other day, he said, “Oh yeah, you guys were enjoying that economic Renaissance because of what we did with Barack Obama.”
David Perdue: (01:06)
Now guys, I’m just a dumb business guy for right over that hill, right there, and I can tell you, that is a lie right out of the pit of hell. Donald Trump, after eight years of Barack Obama and Joe Biden giving us the lowest economic output in U.S. history, and we felt that right here in Georgia, Donald Trump gave us the best economic turnaround in U.S. history, y’all.
David Perdue: (01:36)
Seven and a half million new jobs, highest middle class income ever measured, the lowest unemployment for African Americans, Asian Americans, and Hispanics ever measured. After 800,000 people, under Barack Obama, fell into poverty, under Donald Trump, six and a half million people have lifted themselves out of poverty, y’all. Now that’s America.
David Perdue: (01:57)
But it took Donald Trump, not the bureaucrats in Washington, not some career politician, but it took Donald Trump to build that wall. It took Donald Trump to move that embassy from Tel Aviv to Jerusalem. It took Donald Trump to name 200 new federal judges, including Neil Gorsich and Brett Kavanaugh. And next week, we will know that we will confirm Amy Cony Barrett. Mark my words.
David Perdue: (02:43)
It’s one thing to stand up here and cheer. It’s one thing to stand up and remind everybody. This guy is providential. He didn’t happen by accident. How in the world, in our political system, could Donald J. Trump come on the scene in 2016 and do what he did? Tell me. But God’s watching. He’s watching what we’re going to do right now. What’s at stake is this, and I mean this: It’s our time to stand up to the socialist movement that for the last 100 years has been trying to take over our beloved America.
David Perdue: (03:27)
It didn’t just start in the ’30s. It started long ago, but they believe they have it in their grasp now. They believe these CNN polls and CNBC polls. Oh, let me tell you, they don’t know the big red poll right here. But this is what I believe is a stake. Mark my words. They believe this. That if they get the majority of the Senate and they win the White House, Joe Biden has already said, your taxes are going up, guys.
David Perdue: (03:56)
He’s already said. You know those regulations that those mean old demonic Republicans, all those regulations you pulled back, we’re going to give it back to you. You know that healthcare that we screwed up? We’re going to do it again. But most importantly, guys, they want to stack the Supreme Court. Did you see that? They want to stack the Supreme Court. Now what President Trump has done is brought balance back to the Supreme Court. These aren’t activists. These are constitutionalists. They’re up there to protect you and me.
David Perdue: (04:32)
But the most insidious thing that Chuck Schumer and Joe Biden are trying to perpetrate, and Bernie and Elizabeth and Kamala, or what? Kamala or Kamala or Kamala mala mala, I don’t know. Whatever. They’re trying to perpetrate a lie, and that is that socialism, this radical socialism that they’re trying to sell us, is better than our beloved democracy. We’re not going to let that stand.
David Perdue: (04:57)
And here’s how they’re going to do it. You stack the courts, you change the voting rules in the Senate, and pretty soon you start adding democratic states. You get rid of the electoral college or change it like they can do, and pretty soon you’re talking about a one-party state, guys. That failed in Russia in 1917, it failed in Cuba in 1959, Venezuela today. You can’t find a country in the world where that has ever worked.
David Perdue: (05:19)
Now why in the world would we ever go down that road, particularly after the last 75 years in this country where God has blessed us so greatly that we can come out today and say what we want independent of what that media says? I need you to do one thing. Stand with me, stand with our President, and let’s make damn sure that the road to socialism never runs through the state of Georgia. God bless you guys.'''
    
## 4. KeyBert
from keybert import KeyBERT

kw_model = KeyBERT('distilbert-base-nli-mean-tokens')

def keywords(doc):
    return kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=None)

## 5. Top Words


## Data Structure
def data_structure(summary,t2e,kw,sent):
  # Summary
  obj = {'SpeechSummary':'','emotionData':[],'topword':[],'DoughnutData':{},'WordCloud':''}
  obj['SpeechSummary'] = summary

  # t2e
  emot = list(t2e.keys())
  val = list(t2e.values())
  for em,va in zip(emot, val):
    obj['emotionData'].append({'emotion':em,'value':va})
  
  # Keywords
  for i in range(0,5):
    obj['topword'].append({'primary':kw[i][0]})
  
  # Sent
  del sent['compound']
  obj['DoughnutData']['datasets'] = [{'data':list(sent.values())}]
  obj['DoughnutData']['labels'] = list(sent.keys())

  return obj
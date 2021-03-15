import io, sys, pickle
import numpy as np
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pdb

class word_cloud:
	comm = []
	def __init__(self, communities):
		self.comm = communities

	def make_cloud():
		venues = pickle.load(open('List_Venues.p', 'rb'))
		print(len(venues))
		# stopwords = set(STOPWORDS) 
		# comment_words = "Hi I am Vinayak Vinayak Vinayak Vinayak Vinayak"
  
		# wordcloud = WordCloud(width=800,height=800,
		# 	background_color='white',stopwords=stopwords,
		# 	min_font_size=10).generate(comment_words)

		# plt.figure(figsize = (8, 8), facecolor = None) 
		# plt.imshow(wordcloud) 
		# plt.axis("off") 
		# plt.tight_layout(pad = 0) 
		  
		# plt.show()

	def make_string():
		x = pickle.load()
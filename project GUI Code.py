from tkinter import *
from tkinter import scrolledtext, messagebox, ttk
from PIL import ImageTk, Image
import networkx as nx
import pandas as pd
from networkx.algorithms import community
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from empath import Empath
from networkx.algorithms.community.quality import *
from collections import Counter
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
def ProjectWindow():
 project_button.config(state=DISABLED)
 ProjectGUIWindow()
intro_window = Tk()
intro_window.title("Top_Level1")
introwintop = Toplevel()
introwintop.title('Mini Project')
introwintop.config(background = "red")
logo_frame0 = Frame(introwintop)
img = ImageTk.PhotoImage(Image.open("download.png"), master=introwintop)
logo_label = Label(logo_frame0, image = img)
logo_label.pack()
logo_frame0.pack()
intro_frame1 = Frame(introwintop)
title_label = Label(intro_frame1, text='MINI-PROJECT CSE300\n\n'+'PROJECT TITLE\n' +
'EMOTIONAL ANALYSIS OF THE BEHAVIOUR OF DEPRESSIVE USER
INTERACTIONS ON SOCIAL NETWORKS', height=5, font = ('Times New Roman',16),
relief = 'solid')
title_label.pack(expand=True)
team_members_label = Label(intro_frame1, text = 'TEAM MEMBERS\n' + 'BALAJI
VIGNESH R A A -> 124015013 -> B.TECH - IT\n' + 'KARTHIK T -> 124015046 ->
B.TECH - IT\n' + 'JAYANTH NARAYANAN S -> 124157048 -> B.TECH - CSE-CSBT\n',
height=5, font = ('Times New Roman',16), relief = 'solid')
team_members_label.pack(side=BOTTOM, fill='both', expand=True)
intro_frame1.pack(pady=40)
project_button_frame = Frame(introwintop)
project_button = Button(project_button_frame, text = "PROJECT WINDOW",font = ('Times
New Roman',16),borderwidth=5, command=ProjectWindow)
project_button.pack()
project_button_frame.pack()
project_window.mainloop()
class ProjectGUIWindow:
 def show_network_graph(self):
     self.graph_button.config(state=DISABLED)
     graph_window = Toplevel()
     img = Image.open("Untitled2.png")
     resized_image = img.resize((700,700))
     self.graph_image = ImageTk.PhotoImage(resized_image)
     label = Label(graph_window, image=self.graph_image)
     label.pack()
     graph_window.title("Network Graph Viewer")
     self.edge_df = pd.read_csv('EdgesX.csv')
     self.G = nx.DiGraph()
     for i, row in self.edge_df.iterrows():
         source = row['Source']
         target = row['Target']
         edge_type = row['Type']
         weight = row['Weight']
         self.G.add_edge(source, target, type=edge_type, weight=weight)
     self.graph_created=True

 def cnm(self):
     if not self.graph_created:
         messagebox.showwarning("Warning", "Please create the complex network graph first.")
         return

     self.cnm_button.config(state=DISABLED)
     output_window = Toplevel()
     output_window.title("Community Detection Algorithm Output")
     output_text = scrolledtext.ScrolledText(output_window, wrap=WORD, width=80,height=30, font=('TkDefaultFont', 10))
     output_text.pack(fill='both', expand=True)
     output_text.configure(state='disabled')
     self.communities = community.greedy_modularity_communities(self.G)
     for comm in self.communities:
         if len(comm)>5:
             self.top_communities.append(comm)

     for i,c in enumerate(self.communities):
         output_text.configure(state='normal')
         output_text.insert(END,f"Community{i}:{str(list(c))}\n")
         output_text.configure(state='disabled')
 def emus(self):
     if not self.graph_created:
         messagebox.showwarning("Warning", "Please create the complex network graphfirst.")
         return
     if self.graph_created:
         if not self.top_communities:
             messagebox.showwarning("Warning", "Please run the community detectionalgorithm first.")

             return
         self.Dict={}
         self.Dict_Data = {}

         self.fd = pd.read_excel('Depression_Interaction - Copy.xlsx')
         for j,x in enumerate(self.top_communities):
             community_name = f"community{j+1}"
             inner_Dict={}
             name_list = []
             fSet=x
             for value in fSet:
                 name_list.append(value)

             author_dict = {}
             for author_name in name_list:
                 post_mask = self.fd["Post Author"] == author_name
                 comment_mask = self.fd["Comment Author"] == author_name
                 reply_mask = self.fd["Reply Author"] == author_name
                 self.posts = self.fd.loc[post_mask, "Post"].tolist()
                 self.comments = self.fd.loc[comment_mask, "Comment"].tolist()
                 self.replies = self.fd.loc[reply_mask, "Reply"].tolist()
                 self.comments_and_replies = self.comments + self.replies
                 author_dict[author_name] = {"text": [self.posts,self.comments_and_replies]}
                 Score, Emotions = self.EMUS(self.posts, self.comments_and_replies)
                 inner_Dict[author_name] = [Score, Emotions]
             self.Dict[community_name] = author_dict
             self.Dict_Data[community_name] = inner_Dict


        emus_window = Toplevel()
        emus_window.title("Emotional User Score Window")
        text_widget = Text(emus_window)
        text_widget.pack(expand=True, fill=BOTH)
        table_data = []
        for community, nodes in self.Dict_Data.items():
            for node, data in nodes.items():
                score = data[0]
                emotions = ", ".join(data[1])
                table_data.append([community, node, score, emotions])
        table = tabulate(table_data, headers=["Community", "Node", "Score", "Emotions"])
        text_widget.insert(END, table)
        text_widget.config(state="disabled")
        x_scrollbar = Scrollbar(emus_window, orient=HORIZONTAL)
        x_scrollbar.pack(fill=X, side=BOTTOM)
        text_widget.configure(xscrollcommand=x_scrollbar.set)
        text_widget.config(wrap=NONE)
        x_scrollbar.configure(command=text_widget.xview)
        y_scrollbar = Scrollbar(emus_window, orient=VERTICAL,command=text_widget.yview)
        text_widget.configure(yscrollcommand=y_scrollbar.set)
        y_scrollbar.pack(fill=Y, side=RIGHT)

        button_frame = Frame(emus_window)
        button_frame.pack(pady=10)

        topemoclass_button = Button(button_frame, text="Top Emotions Classification",command=self.show_top_emotions)
        topemoclass_button.pack(side=LEFT, padx=10)
        histo_button = Button(button_frame, text="Histogram",command=self.display_histogram)
        histo_button.pack(side=LEFT, padx=10)
  def show_top_emotions(self):
      top_emotions_window = Toplevel()
      top_emotions_window.title("Top Emotions for each Community")
      text_widget = Text(top_emotions_window)
      text_widget.pack(expand=True, fill=BOTH)
      emotion_dict = {}
      community_sizes = []
      community_avg_scores = []
      for comm_num, comm_data in self.Dict_Data.items():
          emotions = []
          scores = []
      for user, data in comm_data.items():
          emotions.append(data[1])
          scores.append(data[0])
      flat_list = [item for sublist in emotions for item in sublist]
      emotion_dict[comm_num] = flat_list
      community_sizes.append(len(comm_data))
      community_avg_scores.append(sum(scores) / len(scores))
  top_emotions = 10
  results = []
  for idx, (community, emotions) in enumerate(emotion_dict.items()):
      emotion_counts = Counter(emotions)
      top_emotions_list = emotion_counts.most_common(top_emotions)
      top_emotions_only = [emotion[0] for emotion in top_emotions_list]
      results.append([community, community_sizes[idx], community_avg_scores[idx],top_emotions_only])
      table = tabulate(results, headers=["Community", "Size", "Avg Score", "Top Emotions"])
      text_widget.insert(END, table)
      text_widget.config(state="disabled")
      x_scrollbar = Scrollbar(top_emotions_window, orient=HORIZONTAL)
      x_scrollbar.pack(fill=X, side=BOTTOM)
      text_widget.configure(xscrollcommand=x_scrollbar.set)
      text_widget.config(wrap=NONE)
      x_scrollbar.configure(command=text_widget.xview)
      y_scrollbar = Scrollbar(top_emotions_window, orient=VERTICAL,command=text_widget.yview)
      text_widget.configure(yscrollcommand=y_scrollbar.set)
      y_scrollbar.pack(fill=Y, side=RIGHT)
  def display_histogram(self):
      histo_window = Toplevel()
      histo_window.title("Distribution of Users Across Emotion Categories")
      Histo_List = []
      for i, c in self.Dict_Data.items():
          for j, k in c.items():
          Histo_List.append(k[0])
      e = h = s = d = vd = 0

      for i in Histo_List:
          if i <= 1 and i > 0.66:
              e = e + 1
          elif i <= 0.66 and i > 0.26:
              h = h + 1
          elif i <= 0.26 and i > -0.26:
              s = s + 1
          elif i <= -0.26 and i > -0.66:
              d = d + 1
          elif i <= -0.66 and i >= -1:
              vd = vd + 1
  categories = ['Euphoric', 'Happy', 'Sad', 'Depressed', 'Very Depressed']
  number_of_users = [e, h, s, d, vd]
  fig = Figure(figsize=(6,5), dpi=80)
  fig.subplots_adjust(bottom=0.2)
  ax = fig.add_subplot(111)
  bars = ax.hist(categories, weights=number_of_users, bins=len(categories), align='mid')
  ax.set_xticklabels(categories, rotation=45)
  ax.set_yticks(range(0, max(number_of_users) + 10, 15))
  ax.set_ylabel('Number of Users')
  ax.set_xlabel('Categories')
  ax.set_title('User Emotions')
  for i, bar in enumerate(bars[0]):
      ax.text(i, bar + 1, str(int(bar)), ha='center')
  canvas = FigureCanvasTkAgg(fig, master=histo_window)
  canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)



 def metrics_display(self):
     if not self.graph_created:
         messagebox.showwarning("Warning", "Please create the complex network graphfirst.")
     return
     if self.graph_created:
         if not self.communities:
             messagebox.showwarning("Warning", "Please run the community detectionalgorithm first.")
         return
     modularity_score, performance, coverage, assortativity_coeff, avg_clustering, diameter,density, Final_Precision, Final_Recall, f_one_score = self.metrics_calc()
     messagebox.showinfo('Metrics', f'Network Metrics:\nDensity: {density:.4f}\nAverage
Clustering Coefficient: {avg_clustering:.4f}\nAssortativity Coefficient:
{assortativity_coeff:.3f}\nDiameter: {diameter}\nCoverage: {coverage}\nPerformance:
{performance:.3f}\nModularity: {modularity_score:.3f}\n\nCommunity Detection Metrics:\n
Precision: {Final_Precision:.3f}\nRecall: {Final_Recall:.2f}\n F1-Score: {f_one_score:.3f}')

     Net_Deg = Toplevel()
     Net_Deg.title("Network Degree Distribution")
     degree_dist = [j for i, j in self.G.degree()]
     canvas = Canvas(Net_Deg, width=800, height=600)
     canvas.pack()
     plt.hist(degree_dist, bins='auto')
     plt.title("Network Degree Distribution")
     plt.xlabel("Degree")
     plt.ylabel("Frequency of Users")
 for rect in plt.gca().patches:
     height = rect.get_height()
     if height > 0:
        plt.gca().text(rect.get_x() + rect.get_width() / 2, height,f'{int(height)}', ha='center', va='bottom')
     figure = plt.gcf()
     figure.set_size_inches(8, 6)
     figure_canvas = FigureCanvasTkAgg(figure, master=canvas)
     figure_canvas.draw()
     figure_canvas.get_tk_widget().pack()
     figure_canvas.get_tk_widget().configure(borderwidth=0)
     figure_canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
     figure_canvas._tkcanvas.pack(side='top', fill='both', expand=True)
 def metrics_calc(self):
     mod_score = modularity(self.G, self.communities)
     intra_edges = nx.algorithms.community.quality.intra_community_edges(self.G,self.communities)
     non_edges = nx.algorithms.community.quality.inter_community_non_edges(self.G,self.communities)
     total_possible_edg = len(self.G.nodes)*(len(self.G.nodes)-1)
     cgp = (intra_edges + non_edges) / total_possible_edg
     ccg = intra_edges/len(self.G.edges)
     r = nx.degree_assortativity_coefficient(self.G)
     ac = nx.average_clustering(self.G)
     dia = max([max(b.values()) for (a,b) in nx.shortest_path_length(self.G)])
     den = nx.density(self.G)
     truth_communities = tuple(sorted(tuple(c) for c in next(community.centrality.girvan_newman(self.G))))
     gnmod_score = modularity(self.G, truth_communities)

     local_communities = tuple(sorted(tuple(c) for c in community.greedy_modularity_communities(self.G)))

     precision_total = 0
     recall_total = 0
     for local_community in local_communities:
         max_precision = 0
         max_recall = 0
         for truth_community in truth_communities:
             TP = len(set(local_community) & set(truth_community))
             FP = len(local_community) - TP
             FN = len(truth_community) - TP
             precision = TP / (TP + FP)
             recall = TP / (TP + FN)
             if precision > max_precision:
                 max_precision = precision
                 max_recall = recall
         precision_total += max_precision
         recall_total += max_recall
    avg_precision = precision_total / len(local_communities)
    avg_recall = recall_total / len(local_communities)
    f_score = 2*(avg_precision*avg_recall)/(avg_precision+avg_recall)
    return mod_score, cgp, ccg, r, ac, dia, den, avg_precision, avg_recall, f_score



 def EMUS(self,userPosts, userComments):
     sia = SentimentIntensityAnalyzer()
     lexicon = Empath()
     userScore=0
     userEmotions=[]
     if not userPosts:
         sNeg = (sum(sia.polarity_scores(comment)['neg'] for comment in userComments))*-1
         sPos = sum(sia.polarity_scores(comment)['pos'] for comment in userComments)
         EmotionsDict = lexicon.analyze(userComments, normalize=True)
     elif not userComments:
         sNeg = (sum(sia.polarity_scores(post)['neg'] for post in userPosts))*-1
         sPos = sum(sia.polarity_scores(post)['pos'] for post in userPosts)
         EmotionsDict = lexicon.analyze(userPosts, normalize=True)
     else:
         sNeg = (sum(sia.polarity_scores(post)['neg'] for post in userPosts) +sum(sia.polarity_scores(comment)['neg'] for comment in userComments))*-1
         sPos = sum(sia.polarity_scores(post)['pos'] for post in userPosts) +sum(sia.polarity_scores(comment)['pos'] for comment in userComments)

         EmotionsDict = lexicon.analyze(userPosts + userComments, normalize=True)
     for i,c in enumerate(EmotionsDict):
         if EmotionsDict[c]>0.0 or EmotionsDict[c]<0.0:
             userEmotions.append(c)

     userScore = sNeg + sPos
     return (userScore, userEmotions)

 def __init__(self):
     self.communities=None
     self.top_communities=None
     self.Dict=None

     self.Dict_Data=None
     self.graph_created=False
     self.edge_df=None
     self.G=None
     self.top_communities=[]
     self.projectwindow = Tk()
     self.projectwindow.title("Project Window")
     self.projectwindow.config(bg='red')
     intro_frame1 = Frame(self.projectwindow)
     title_label = Label(intro_frame1, text='MINI-PROJECT CSE300\n\n'+'PROJECTTITLE\n' + 'EMOTIONAL ANALYSIS OF THE BEHAVIOUR OF DEPRESSIVE USERINTERACTIONS ON SOCIAL NETWORKS',height=5, font = ('Times New Roman',16), relief = 'solid')
     title_label.pack(expand=True)
     intro_frame1.pack(pady=40)

     options_frame = Frame(self.projectwindow, bg='blue',relief='sunken',highlightthickness=5)
     self.graph_button = Button(options_frame,text='DISPLAY COMPLEX NETWORKGRAPH', font=('Consolas',16), command=self.show_network_graph)
     self.graph_button.pack(side=TOP,expand=True,fill="both")
     self.cnm_button = Button(options_frame,text='RUN COMMUNITY DETECTION ALGORITHM', font=('Consolas',16), command=self.cnm)
     self.cnm_button.pack(side=LEFT)
     self.emus_button = Button(options_frame,text='RUN EMOTIONAL ANALYSIS',font=('Consolas',16), command=self.emus)
     self.emus_button.pack(side=LEFT)
     self.metrics_button = Button(options_frame,text='ASSESS GRAPH STRUCTURE',font=('Consolas',16), command=self.metrics_display)
     self.metrics_button.pack(side=LEFT)
     options_frame.pack()

     self.projectwindow.mainloop()

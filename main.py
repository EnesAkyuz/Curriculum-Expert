import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import telebot
import networkx as nx
import matplotlib.pyplot as plt

# KEYS
ORG_KEY = os.environ.get("ORG_KEY")
PROJECT_ID = os.environ.get("PROJECT_ID")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

# CLIENTS
bot = telebot.TeleBot(TELEGRAM_TOKEN)
client = OpenAI(
  organization= ORG_KEY,
  project= PROJECT_ID,
)

# Sorry for the fact that I did not want to initiate an entire SQLite server for this
# I am not deploying the app anyways
user_session_database = {}

# Prepare and vectorize the dataset for future search
courses = pd.read_csv("Online_Courses.csv")
courses = courses.drop_duplicates(subset=['Short Intro'])
courses.fillna(value={'Skills': '', 'Short Intro': ''}, inplace=True)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(courses['Skills'])

# Starting message on my Telegram Bot
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Hello, t'is I, CoursExpert. I recommend you online-courses and learning pathways! Can you tell me what do you want to learn about today?")

# Search user's query to pull out the top10 courses based on the cosine similarity
# Thanks to CS113 for teaching me what cosine similarity is here
@bot.message_handler(func=lambda message: True)
def handle_query(message):

    # This is to further use the top10 courses picked
    chat_id = message.chat.id

    # sort the dataset according to the cosine similarity with user's query
    query = message.text
    vectorized_query = vectorizer.transform([query])
    rankings = cosine_similarity(vectorized_query, vectorizer.transform(courses['Skills'])).flatten()
    courses['cos_sim'] = rankings
    top10 = courses.sort_values(by='cos_sim', ascending=False).head(10)

    # record the database into user's query
    user_session_database[chat_id] = top10

    # Create a response message with the top course titles and URLs
    response_message = "\nHere are the top courses for your interest:\n"
    for i, row in top10.iterrows():
        response_message += f"\n{row['Title']} \n {row['URL']}\n"

    # Send the response back to the user
    bot.reply_to(message, response_message)

    # Go to the next question for designing a graph and a path
    question2(message)

@bot.message_handler(commands=['find_path'])
def ask_for_path_limit(message):
    msg = bot.send_message(message.chat.id, "Please enter the maximum number of courses you would like to undertake:")
    bot.register_next_step_handler(msg, process_path_request)

def process_path_request(message):
    try:
        max_courses = int(message.text)
        chat_id = message.chat.id
        if chat_id in user_session_database and 'graph' in user_session_database[chat_id]:
            G = user_session_database[chat_id]['graph']
            path = longest_path_within_limit(G, max_courses)
            if path:
                course_titles = [G.nodes[node]['title'] for node in path]
                response_message = "Recommended path for " + str(max_courses) + " courses:\n" + " -> ".join(course_titles)
            else:
                response_message = "No valid path found within the limit of " + str(max_courses) + " courses."
        else:
            response_message = "Graph not available. Please generate the network first."
    except ValueError:
        response_message = "Please enter a valid number of courses."
    bot.send_message(chat_id, response_message)

def longest_path_within_limit(G, max_courses):
    longest_path = []
    # Iterate over all pairs of nodes to find the longest path from any node to any other
    for start_node in G.nodes():
        for target_node in G.nodes():
            if start_node != target_node:
                paths = list(nx.all_simple_paths(G, source=start_node, target=target_node, cutoff=max_courses - 1))
                # Find the longest path from the current list of paths and compare it with the overall longest
                if paths:
                    current_longest = max(paths, key=len)
                    if len(current_longest) > len(longest_path):
                        longest_path = current_longest
    if longest_path:
        # Convert the path of edges to a path of node titles
        node_path = [G.nodes[node]['title'] for node in longest_path]
        return node_path
    return longest_path

def question2(message):
    markup = telebot.types.InlineKeyboardMarkup()
    button = telebot.types.InlineKeyboardButton("Click if yes!", callback_data='network_printer')
    markup.add(button)
    bot.send_message(message.chat.id, "Do you want to have the transition network?", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data == 'network_printer')
def handle_query_details(call):
    chat_id = call.message.chat.id

    # Check if the previous message exists basically
    if chat_id in user_session_database:
        top_courses = user_session_database[chat_id]

        # Generate and send the network diagram here
        diagram_path = generate_diagram(top_courses, chat_id)

        # send the diagram as a photo, I needed to save it beforehand
        # I will handle that in the generate diagram function
        with open(diagram_path, 'rb') as photo:
            bot.send_photo(chat_id, photo)
            # Prompt the user to enter the course limit after showing the diagram
            msg = bot.send_message(chat_id, "If you want me to create a path, please enter the number of courses you want me to take (1-10):")
            bot.register_next_step_handler(msg, process_path_request)
    # In case of anything going bad
    else:
        bot.send_message(chat_id, "Sorry, I couldn't retrieve the network.")
    bot.answer_callback_query(call.id)  # to ensure that the loading icon on the button goes away

# Network diagram generator
def generate_diagram(top_courses, chat_id):
    # This function should generate a network diagram and return the file path
    # Use matplotlib and networkx to create and save the diagram
    # Return the path to the saved image file
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(top_courses['Skills'])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    G = nx.Graph()
    course_titles = [' '.join(title.split()[:3]) for title in top_courses['Title'].tolist()]

    # Add nodes with course titles
    for i, title in enumerate(course_titles):
        G.add_node(title, title=title)

    # Add edges based on similarity scores from the matrix
    for i in range(len(course_titles)):
        for j in range(i + 1, len(course_titles)):

            # At this point, the graph is extremely crowded
            # So, I will just add a threshold to avoid this
            if similarity_matrix[i][j] > 0.4:
                G.add_edge(course_titles[i], course_titles[j], weight=round(similarity_matrix[i][j], 2))

    user_session_database[chat_id] = {'top_courses': top_courses, 'graph': G}

    # That was the only way I found out to avoid the nodes from getting on top of each other
    # This is credited to stackoverflow mainly
    pos = nx.spring_layout(G, iterations=30)

    # I wanted to make the nodes bigger
    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color='lightgreen', alpha=0.6)

    # Draw edges
    edge_widths = [5 * G[u][v]['weight'] for u, v in G.edges]  # Scale edge widths
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='black')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_family='sans-serif')

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()},
                                 font_color='red')

    # Save the diagram to a file
    diagram_path = 'path_to_network_diagram.png'
    plt.savefig(diagram_path)
    plt.close()
    return diagram_path



if __name__ == '__main__':
    bot.polling()


"""
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)
print(completion.choices[0].message.content)
"""
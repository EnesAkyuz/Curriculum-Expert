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
    bot.reply_to(message, "Hello, t'is I, CoursExpert. I recommend you online-courses and learning pathways! Can you tell me what do you want to learn about and be as detailed as possible in your description?")

# Search user's query to pull out the top10 courses based on the cosine similarity
# Thanks to CS113 for teaching me what cosine similarity is here
@bot.message_handler(func=lambda message: True)
def handle_query(message):

    # This is to further use the top10 courses picked
    chat_id = message.chat.id

    # sort the dataset according to the cosine similarity with user's query
    # THIS IS BASED ON SKILLS
    query = message.text
    v_query = vectorizer.transform([query])
    rankings = cosine_similarity(v_query, vectorizer.transform(courses['Skills'])).flatten()

    # Add this to the real dataframe as a metric
    # Sort it in ascending order and take the first 10 values to see
    # Which values are the more "cosine-similar" ones
    courses['cos_sim'] = rankings
    top10 = courses.sort_values(by='cos_sim', ascending=False).head(10)

    # record the database into user's query, I am going to use this later
    user_session_database[chat_id] = top10

    # Create a response message with the top course titles and URLs
    response_message = "\nHere are the top courses for your interest:\n"
    # Iterate over rows and append to the big resposne
    for i, row in top10.iterrows():
        response_message += f"\n{row['Title']} \n {row['URL']}\n"

    # Send the response back to the user
    bot.reply_to(message, response_message)

    # Go to the next question for designing a graph and a path
    question2(message)


def process_path_request(message):

    # Have a measure for users entering something other than numbers
    # We only like numbers here haha!
    try:
        # record the number, chat_id and get the user's graph from the recorded database
        max_courses = int(message.text)
        chat_id = message.chat.id
        user_dict = user_session_database[chat_id]

        # check if the user created the network and the graph, otherwise throw an error response
        if chat_id in user_session_database and 'graph' in user_session_database[chat_id]:

            # If user created these, get the longest path
            course_graph = user_session_database[chat_id]['graph']
            path = longest_path_within_limit(course_graph, max_courses)

            # When the path exists, proceed to return it to the user
            if path:
                # Ok this part is a little makeshift because I did not know how to
                # keep the urls with the titles within the graph
                course_titles = [course_graph.nodes[node]['title'] for node in path]
                urls = []

                # So, I just wrote a for loop!
                for title in course_titles:
                    top_courses = user_dict["top_courses"]
                    course_list = top_courses[top_courses["Title"] == title]
                    for url in course_list["URL"]:
                        urls.append(url)
                # If everything goes well, return the *ornamented* ChatGPT improved message to the user
                response_message = improve_message_GPT("Recommended path for " + str(max_courses) + " courses:\n" + " -> ".join(course_titles) + "with the respective urls:\n" + "\n".join(urls))
            else:
                response_message = "No path found with the given number of " + str(max_courses) + " courses."
        else:
            response_message = "Minervan! Follow the instructions, you have not generated a graph yet!"
    except ValueError:
        response_message = "Please only enter the number and nothing else!"
    bot.send_message(chat_id, response_message)

def longest_path_within_limit(course_graph, max_courses):
    longest_path = []

    # This is basically brute force longest path since I only had 10 nodes after filtering
    # Iterate over every path in existence with the given cutoff
    # Then store the longest path
    # Since edges are built with cosine similarity
    # This path will take the most similar courses and smoothest transitions in terms of skills
    # into consideration
    for start in course_graph.nodes():
        for target in course_graph.nodes():
            if start != target:
                # this is a fancy trick I learned from GPT
                paths = list(nx.all_simple_paths(course_graph, source=start, target=target, cutoff=max_courses - 1))
                # Find the longest path from the current list of paths and compare it with the overall longest
                if paths:
                    current_longest = max(paths, key=len)
                    if len(current_longest) > len(longest_path):
                        longest_path = current_longest
    if longest_path:
        # Convert the path of edges to a path of node titles
        node_path = [course_graph.nodes[node]['title'] for node in longest_path]
        return node_path
    return longest_path

# I did not want to provide the user with a choice here
# However, since I was not sure if this feature should have stayed
# I just kept the choice button in
def question2(message):
    markup = telebot.types.InlineKeyboardMarkup()
    button = telebot.types.InlineKeyboardButton("Click here!", callback_data='network_printer')
    markup.add(button)
    bot.send_message(message.chat.id, "I can also create a transition network in which, the proximities of the nodes will symbolize how similar two subjects are in terms of skills.", reply_markup=markup)

# Upon the call from question2, handle how to print the network.
@bot.callback_query_handler(func=lambda call: call.data == 'network_printer')
def handle_query_details(call):
    # Record chat id since I am going to use this to record the topchoices to the user's session
    chat_id = call.message.chat.id

    # Check if the previous message exists basically
    if chat_id in user_session_database:
        top_courses = user_session_database[chat_id]

        # Generate and send the network diagram here, I'll write this function seperately
        diagram_path = generate_diagram(top_courses, chat_id)

        # send the diagram as a photo, I needed to save it beforehand
        # I will handle that in the generate diagram function
        with open(diagram_path, 'rb') as photo:
            bot.send_photo(chat_id, photo)
            # After showing the diagram, I need to prompt them to go to the next question
            # I found no better way of building it in Telegram
            msg = bot.send_message(chat_id, "If you want me to create a path, please enter the number of courses you want me to take (1-10):")
            bot.register_next_step_handler(msg, process_path_request)
    # If something fails on the way
    else:
        bot.send_message(chat_id, "Sorry Minervan! Could not find the graph!")

    # This is a funny bit of line but apparently
    # If we do not close the call, the loading screen stays there forever
    bot.answer_callback_query(call.id)

# Network diagram generator
def generate_diagram(top_courses, chat_id):

    # vectorize the dataset according to skills and put it into a matrix
    # then get the cosine similarity matrix
    vector_creator = TfidfVectorizer()
    vectorized_skills = vector_creator.fit_transform(top_courses['Skills'])
    similarity_matrix = cosine_similarity(vectorized_skills)

    course_graph = nx.Graph()
    course_titles = top_courses['Title'].tolist()

    # Add nodes with course titles
    for i, title in enumerate(course_titles):
        course_graph.add_node(title, title=title)

    # Add edges based on similarity scores from the matrix
    for i in range(len(course_titles)):
        for j in range(i + 1, len(course_titles)):

            # At this point, the graph is extremely crowded
            # So, I will just add a threshold to avoid this
            if similarity_matrix[i][j] > 0.4:
                course_graph.add_edge(course_titles[i], course_titles[j], weight=round(similarity_matrix[i][j], 2))

    # I need to record this to further use this one in the pathfinder
    user_session_database[chat_id] = {'top_courses': top_courses, 'graph': course_graph}

    # That was the only way I found out to avoid the nodes from getting on top of each other
    # This is credited to stackoverflow mainly
    pos = nx.spring_layout(course_graph, iterations=30)
    nx.draw_networkx_nodes(course_graph, pos, node_size=4000, node_color='lightgreen', alpha=0.6)
    edge_widths = [5 * course_graph[u][v]['weight'] for u, v in course_graph.edges]  # Scale edge widths
    nx.draw_networkx_edges(course_graph, pos, width=edge_widths, edge_color='black')

    # Draw labels
    nx.draw_networkx_labels(course_graph, pos, font_size=8, font_weight='bold', font_family='sans-serif')

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(course_graph, 'weight')
    nx.draw_networkx_edge_labels(course_graph, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()},
                                 font_color='red')

    # Save the diagram to a file
    diagram_path = 'path_to_network_diagram.png'
    plt.savefig(diagram_path)
    plt.close()
    return diagram_path

# I use GPT just to make the end message a bit more fancy and understandable to the standard reader
def improve_message_GPT(message):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are responsible for taking this message in and presenting this online course path in a nice way to the user with the urls of the courses, alongside the transferable skills between these courses and what makes this a good path!"},
            {"role": "user", "content": message}
        ]
    )
    return completion.choices[0].message.content

# Run the bot
if __name__ == '__main__':
    bot.polling()


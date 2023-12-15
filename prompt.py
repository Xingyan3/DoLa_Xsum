import random

def create_demo_text(n_shot=8, shuffle=False):
    document, summary = [], []
    document.append(
        "23 October 2015 Last updated at 17:44 BST It's the highest rating a tropical storm can get and is the first one of this magnitude to hit mainland Mexico since 1959. But how are the categories decided and what do they mean? Newsround reporter Jenny Lawrence explains.")
    summary.append(
        "Hurricane Patricia has been rated as a category 5 storm.")

    document.append(
        "The announcement ends months of uncertainty for Cornish Language Partnership staff whose contracts had been due to end. Local government minister Andrew Stunnell said the three-year funding package for the service would help make sure the language survived. But he warned that long term funding should come from Cornwall.")
    summary.append(
        "The government is spending nearly £400,000 to help save the Cornish language.")

    document.append(
        "Charminster bridge was previously untouchable due to its historic status, but authorities agreed its small arches restricted the flow of the River Cerne. English Heritage will now allow the 16th Century bridge to be replaced with a new one that has bigger arches. The bridge had been blamed for nearly wrecking a nearby grade I-listed church during the January 2014 floods.")
    summary.append(
        "Historic Harminster bridge to be replaced for better river flow, safeguarding nearby church.")

    document.append(
        "Riding shotgun, Mrs Obama sang along to hits by Beyonce and Stevie Wonder - although her security limited the drive to the White House compound. Mrs Obama confessed she had only ridden in the passenger's seat of a car once in the last seven years. Corden began hosting CBS's The Late, Late Show in March last year.")
    summary.append(
        "Mrs. Obama sings in car with Corden at White House.")

    document.append(
        "Mills and Clark improved on the silver they won in London by taking Olympic gold in the women's 470 event in Rio. Four-time world champion Giles Scott, who won Finn gold at Rio 2016, was nominated for the men's award but it went to Argentina's Santiago Lange.")
    summary.append(
        "Mills and Clark win gold in Rio; Lange receives men's award.")

    document.append(
        "In recent years, advancements in artificial intelligence have led to significant developments in various fields, ranging from healthcare to automotive industries. One of the major breakthroughs is the emergence of autonomous driving technology. This technology uses a combination of sensors, cameras, and advanced algorithms to navigate and operate vehicles without human intervention. The potential benefits include reduced traffic accidents, improved traffic flow, and lower transportation costs. However, the technology also faces challenges such as regulatory hurdles, safety concerns, and the need for extensive testing and data collection to ensure reliability and public trust.")
    summary.append(
        "AI-driven autonomous driving promises efficiency but faces safety and regulatory challenges.")

    document.append(
        "The global economy has been facing unprecedented challenges due to the COVID-19 pandemic. The widespread lockdowns and social distancing measures have led to a significant slowdown in economic activities across the world. Many businesses have been forced to shut down, resulting in massive job losses and financial instability. Governments around the world have been implementing various fiscal policies and stimulus packages to mitigate the economic impact. The pandemic has also accelerated the shift towards digitalization, as more people are working from home and relying on digital platforms for daily operations.")
    summary.append(
        "COVID-19's impact on the global economy.")

    # randomize order of the examples ...
    index_list = list(range(len(document)))
    if shuffle:
        random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        demo_text += "Document: " + document[i] + "\nSummary: " + summary[i] + " " + "\n\n"

    return demo_text


def build_prompt(input_text, n_shot, shuffle):
    demo = create_demo_text(n_shot, shuffle)
    input_text_prompt = demo + "Document: " + input_text + "\n" + "Summary: "
    return input_text_prompt
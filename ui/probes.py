"""Hand-written live-style probe tickets (copied from the prototype; not dataset rows).

Used only by scripts/build_ui_report.py: IN_DOMAIN places the burst threshold just above the similarity of
*distinct* issues; both sets report how well sigma separates real tickets from off-topic ones.
"""

IN_DOMAIN = [
    "Outlook keeps crashing when I open attachments since the latest update.",
    "Can't log in to the customer portal, it says my account is locked.",
    "We were charged twice for the October invoice, please check.",
    "How do I export my project data to CSV?",
    "Our website is loading extremely slowly since this morning.",
    "Please add two new users to our team license.",
    "The mobile app shows a sync error and doesn't save my changes.",
    "Is there an API for integrating your CRM with our ERP system?",
    "My laptop can't connect to the office Wi-Fi anymore.",
    "I'd like to return a damaged monitor I received yesterday.",
    "Need help configuring two-factor authentication for our admin accounts.",
    "Our backups failed last night with a disk full error.",
    "Ich kann mich seit heute nicht mehr in mein Konto einloggen.",
    "Die Rechnung für November enthält einen falschen Betrag.",
    "Wie kann ich mein Passwort zurücksetzen?",
    "Unsere Website ist seit dem Update sehr langsam.",
    "Bitte schalten Sie für unseren neuen Mitarbeiter einen Zugang frei.",
    "Die Synchronisation zwischen App und Desktop funktioniert nicht.",
    "Ich möchte ein defektes Gerät zurückschicken und umtauschen.",
    "Gibt es eine Schnittstelle zu SAP?",
    "Der Server meldet ständig Speicherfehler.",
    "Unser Drucker zeigt nach dem Firmware-Update einen Fehler an.",
    "Wir möchten unser Abonnement auf den Business-Tarif upgraden.",
    "Seit gestern erhalten wir keine E-Mails mehr von externen Absendern.",
]

OUT_OF_DOMAIN = [
    "What's a good recipe for banana bread?",
    "Can you write me a poem about the ocean?",
    "Who won the football world cup in 2014?",
    "Help me with my calculus homework: integrate x^2 sin x",
    "What is the weather tomorrow in Boston?",
    "Recommend a good science fiction novel for my vacation.",
    "How do I train my puppy to sit?",
    "Translate 'good morning' into Japanese please.",
    "My neighbour's tree is blocking my driveway, what are my rights?",
    "Plan a 5-day trip to Lisbon for two people.",
    "What are the symptoms of the flu and should I see a doctor?",
    "Tell me a joke about cats.",
    "Wie backe ich einen Apfelkuchen?",
    "Schreib mir ein Gedicht über den Herbst.",
    "Wer hat die Bundestagswahl 2021 gewonnen?",
    "Wie wird das Wetter morgen in München?",
    "Kannst du mir bei meinen Mathe-Hausaufgaben helfen? Was ist die Ableitung von x hoch drei?",
    "Empfiehl mir einen guten Krimi.",
    "Wie bringe ich meinem Hund bei, Pfötchen zu geben?",
    "Was sind die besten Sehenswürdigkeiten in Rom?",
    "Wie lange muss ein Ei kochen?",
    "Erzähl mir einen Witz.",
]

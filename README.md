# SlackBotApp
You can use this code to auto generate realistic conversations amongst users.

There are 3 things you need: a slack-user token, a member user id, a channel id, and a hugging face token. 


Heres how to get the slack-user token(Since this program involves two different accounts, you must repeat this step twice and get two different codes for each respective user account):

 

Search up Slack API then go to Your Apps.

Click "Create New App" → "From scratch".

Give it a name (e.g., ConversationSimulator) and select your Slack workspace.

Click Create App.

2️⃣ Add Bot Permissions

We need to give the bot permission to read and send messages.

In your app’s settings, go to "OAuth & Permissions".

Under USER Token Scopes(Make sure it is user and not bot), click "Add an OAuth Scope" and add:

chat:write → Allows bot to send messages

channels:history → Lets bot read public channel messages

groups:history → Lets bot read private channel messages

channels:read

im:write

Scroll up and click "Install App to Workspace".

Click Allow when prompted.

3️⃣ Get Your Bot Token

After installing, you’ll see  User OAuth Token (starts with xoxp-).






Also, you need a Member-ID(Also needs to be repeated for two different Slack accounts): 




Right-click the user’s name (in the sidebar or in a message).

Select View profile.

Click the three dots (…) in the top right.

Choose Copy member ID.

You’ll get something like:

U123ABC45






Next, you need to get your CHANNEL-ID:





Open Slack on your browser or desktop app.

Click on the channel you want your bot to post in.

Click the channel name at the top to open channel details.

Look for “Copy Link” (it might be in a dropdown menu).

Paste the link somewhere — it will look like this:

https://app.slack.com/client/T01234567/C9876543210


The part starting with C (or sometimes G for private channels) is the channel ID.

Example: C9876543210






Heres How to Get Your Hugging Face Token:




Create a Hugging Face Account

Go to https://huggingface.co/join and sign up (free).

Log In

Once logged in, click your profile icon (top-right).

Select Settings.

Generate an Access Token

In the Access Tokens section, click New token.

Name it something like "slack-bot" and select Read permissions (unless you need write access).

Click Generate token.

Copy Your Token

It will look like this: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


Finally, go to the python code file and copy and paste your tokens into their respective locations.

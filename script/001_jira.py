from jira import JIRA

# JIRA server and credentials
jira_server = 'https://your-jira-instance.atlassian.net'
jira_user = 'gjih@novonordisk.com'
jira_token = 'your-api-token'  # Get this from your Atlassian account

jira = JIRA(server=jira_server, basic_auth=(jira_user, jira_token))

issue_keys = ['TEST-13', 'TEST-14', 'TEST-15', 'TEST-16', 'TEST-17']
comment = "Blocked for now. Waiting for approval."

for key in issue_keys:
    jira.add_comment(key, comment)
    print(f"Comment added to {key}")

agentTracker = []

def generateTrackerObject(eventName:str, messages):
    messagesLen = len(messages)
    return {
        "node": eventName,
        "decisionNode": True if "conditional_edge" in eventName else False,
        "lastMessageType": messages[messagesLen-1].type,
        "messagesCount": messagesLen
    }


def updateTracker(eventName, messages):
    trackerObject = generateTrackerObject(eventName, messages)
    if not messages:
        return [trackerObject]
    
    agentTracker.append(trackerObject)
    return agentTracker
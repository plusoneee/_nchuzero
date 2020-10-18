from datetime import datetime

class Memory:
    def __init__(self, title, location):
        self.title = title
        self.timestamp = datetime.now()
        self.location = location

    def to_dictionary(self):
        m_dict = dict()
        m_dict['title'] = self.title
        m_dict['timestamp'] = self.timestamp.strftime("%d-%b-%Y")
        m_dict['location'] = self.location
        return m_dict
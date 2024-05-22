import React from 'react';

const ChatHistory = ({ history }) => {
  return (
    <div className="chat-history">
      {history.map((msg, index) => (
        <div key={index} className="history-message">
          <div className="user">{msg.user}</div>
          <div className="bot">{msg.bot}</div>
        </div>
      ))}
    </div>
  );
};

export default ChatHistory;

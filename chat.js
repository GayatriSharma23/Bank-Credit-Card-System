import React from 'react';

const Chat = ({ messages, loading }) => {
  return (
    <div className="chat">
      {messages.map((msg, index) => (
        <div key={index} className={msg.type}>
          <div className="message">
            {msg.type === 'user' && (
              <div className="user-message">
                <div className="user-icon">U</div>
                <div className="user-content">{msg.content}</div>
              </div>
            )}
            {msg.type === 'bot' && (
              <div className="bot-message">
                <div className="bot-icon"></div>
                <div className="bot-content">{msg.content}</div>
              </div>
            )}
          </div>
        </div>
      ))}
      {loading && (
        <div className="loading-dots">
          <span>.</span>
          <span>.</span>
          <span>.</span>
        </div>
      )}
    </div>
  );
};

export default Chat;

import React from 'react';

const Chat = ({ messages, loading }) => {
  return (
    <div className="chat">
      {messages.map((msg, index) => (
        <div key={index} className="message">
          <div className="user">{msg.user}</div>
          <div className="bot">{msg.bot}</div>
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

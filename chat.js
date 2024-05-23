import React from 'react';

const Chat = ({ messages, loading }) => {
  return (
    <div className="chat">
      {messages.map((msg, index) => (
        <div key={index} className="message">
          <div className="user">{msg.user}</div>
          <div className="bot">
            {msg.bot}
            <img src="/laptop-icon.png" alt="Laptop" className="laptop-icon" />
          </div>
        </div>
      ))}
      {loading && (
        <div className="loading-dots">
          <span>.</span><span>.</span><span>.</span>
        </div>
      )}
    </div>
  );
};

export default Chat;

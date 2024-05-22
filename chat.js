import React from 'react';

const Chat = ({ messages }) => {
  return (
    <div className="chat">
      {messages.map((msg, index) => (
        <div key={index} className="message">
          <div className="user">{msg.user}</div>
          <div className="bot">{msg.bot}</div>
        </div>
      ))}
    </div>
  );
};

export default Chat;

import React from 'react';
import SQLButton from './SQLButton';
import VisualButton from './VisualButton';

const Chat = ({ messages, loading, sqlQuery, visualData }) => {
  return (
    <div className="chat">
      {messages.map((msg, index) => (
        <div key={index} className={msg.type === 'user' ? 'user-message' : 'bot-message'}>
          <div className="message">
            {msg.type === 'user' && (
              <div className="user-message">
                <div className="user-content">{msg.content}</div>
              </div>
            )}
            {msg.type === 'bot' && (
              <div className="bot-message">
                <div className="bot-icon"></div>
                <div className="bot-content">
                  {msg.content}
                  {sqlQuery && <SQLButton query={sqlQuery} />}
                  {visualData && <VisualButton visualData={visualData} />}
                </div>
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

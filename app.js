import React, { useState } from 'react';
import Chat from './Chat';
import ChatInput from './ChatInput';
import ChatHistory from './ChatHistory';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleNewChat = () => {
    setMessages([]);
  };

  const handleSendMessage = async (message) => {
    setLoading(true);
    const response = await fetch('/get', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ msg: message }),
    });
    const data = await response.text();  // Assuming the response is plain text
    setMessages([...messages, { user: message, bot: data }]);
    setHistory([...history, { user: message, bot: data }]);
    setLoading(false);
  };

  const handleShowHistory = () => {
    setShowHistory(!showHistory);
  };

  return (
    <div className="app">
      <div className="header">
        <h1 className="app-title">PIA</h1>
        <div className="buttons">
          <button className="new-chat-button" onClick={handleNewChat}>+ New Chat</button>
          <button className="history-button" onClick={handleShowHistory}>Chat History</button>
        </div>
      </div>
      {showHistory ? (
        <ChatHistory history={history} />
      ) : (
        <>
          <Chat messages={messages} loading={loading} />
          <ChatInput onSendMessage={handleSendMessage} />
        </>
      )}
    </div>
  );
};

export default App;

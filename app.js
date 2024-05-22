import React, { useState } from 'react';
import Chat from './Chat';
import ChatInput from './ChatInput';
import ChatHistory from './ChatHistory';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  const handleNewChat = () => {
    setMessages([]);
  };

  const handleSendMessage = async (message) => {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    setMessages([...messages, { user: message, bot: data.response }]);
    setHistory([...history, { user: message, bot: data.response }]);
  };

  const handleShowHistory = () => {
    setShowHistory(!showHistory);
  };

  return (
    <div className="app">
      <div className="header">
        <button onClick={handleNewChat}>New Chat</button>
        <button onClick={handleShowHistory}>Chat History</button>
      </div>
      {showHistory ? (
        <ChatHistory history={history} />
      ) : (
        <>
          <Chat messages={messages} />
          <ChatInput onSendMessage={handleSendMessage} />
        </>
      )}
    </div>
  );
};

export default App;

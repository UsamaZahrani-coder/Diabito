import React, { useEffect, useState } from 'react';
import { View, Text, TextInput, ScrollView, TouchableOpacity, ActivityIndicator, StyleSheet, SafeAreaView, Image } from 'react-native';
import { router } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [profilePicture, setProfilePicture] = useState<string | null>(null);
  const API_URL = "http://192.168.0.111:5000";

  useEffect(() => {
    const loadProfilePicture = async () => {
      try {
        const token = await AsyncStorage.getItem('token');
        if (token) {
          const response = await fetch(`${API_URL}/get-profile`, {
            headers: { Authorization: `Bearer ${token}` }
          });
          const data = await response.json();
          if (data.profile_picture) {
            const profilePicUrl = `${API_URL}/static/uploads/${data.profile_picture}`;
            console.log('Profile picture URL:', profilePicUrl); // Debug log
            setProfilePicture(profilePicUrl);
          } else {
            console.log('No profile picture found in data'); // Debug log
          }
        }
      } catch (error) {
        console.error('Error loading profile picture:', error);
      }
    };
    loadProfilePicture();
  }, []);

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage: Message = {
      role: 'user',
      content: inputMessage.trim()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': '',
          'HTTP-Referer': 'http://localhost:3000',
          'X-Title': 'Diabito Chat'
        },
        body: JSON.stringify({
          messages: [...messages, userMessage].map(msg => ({
            role: msg.role === 'user' ? 'user' : 'assistant',
            content: msg.content
          })),
          model: 'deepseek/deepseek-r1:free',
          temperature: 0.7,
          max_tokens: 1000
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();
      if (!data.choices || !data.choices[0] || !data.choices[0].message) {
        throw new Error('Invalid response format from API');
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: data.choices[0].message.content
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error communicating with the AI service. Please try again.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={{ flex: 1 }}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>🐋 AI Chat</Text>
        </View>
        <ScrollView style={styles.messagesContainer}>
          {messages.map((message, index) => (
            <View
              key={index}
              style={[
                styles.messageRow,
                message.role === 'user' ? styles.userMessageRow : styles.assistantMessageRow
              ]}
            >
              {message.role === 'user' && (
                <View style={styles.profilePictureContainer}>
                  {profilePicture ? (
                    <Image
                      source={{ uri: profilePicture }}
                      style={styles.profilePicture}
                      onError={(error) => console.error('Image loading error:', error.nativeEvent.error)}
                    />
                  ) : (
                    <View style={[styles.profilePicture, styles.profilePicturePlaceholder]}>
                      <Text style={styles.profilePicturePlaceholderText}>?</Text>
                    </View>
                  )}
                </View>
              )}
              <View
                style={[
                  styles.messageBox,
                  message.role === 'user' ? styles.userMessage : styles.assistantMessage
                ]}
              >
                <Text style={[styles.messageText, message.role === 'user' ? styles.userMessageText : styles.assistantMessageText]}>
                  {message.content}
                </Text>
              </View>
            </View>
          ))}
          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#007AFF" />
            </View>
          )}
        </ScrollView>

        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            value={inputMessage}
            onChangeText={setInputMessage}
            placeholder="Type your message..."
            multiline
            returnKeyType="send"
            onSubmitEditing={sendMessage}
          />
          <TouchableOpacity
            style={[styles.sendButton, !inputMessage.trim() && styles.sendButtonDisabled]}
            onPress={sendMessage}
            disabled={!inputMessage.trim()}
          >
            <Text style={styles.sendButtonText}>Send</Text>
          </TouchableOpacity>
        </View>
      </View>
      <View style={styles.bottomNav}>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => router.push('/')}
        >
          <Image 
            source={require('../icons/home.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => router.push('/calculator')}
        >
          <Image 
            source={require('../icons/calculator.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => router.push('/chat')}
        >
          <Image 
            source={require('../icons/chat_cliked.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => router.push('/profile')}
        >
          <Image 
            source={require('../icons/Profile.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f6fa'
  },
  header: {
    backgroundColor: '#fff',
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
    alignItems: 'center'
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#2c3e50'
  },
  messagesContainer: {
    flex: 1,
    padding: 10,
  },
  messageBox: {
    maxWidth: '80%',
    padding: 12,
    marginVertical: 6,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#007AFF',
    marginLeft: '20%',
  },
  assistantMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#E8E8E8',
    marginRight: '20%',
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  loadingContainer: {
    padding: 12,
    alignItems: 'center',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 12,
    backgroundColor: '#FFFFFF',
    borderTopWidth: 1,
    borderTopColor: '#E5E5EA',
    alignItems: 'center',
  },
  input: {
    flex: 1,
    marginRight: 12,
    padding: 12,
    backgroundColor: '#F8F8F8',
    borderRadius: 20,
    fontSize: 16,
    maxHeight: 100,
    color: '#000000',
  },
  sendButton: {
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#007AFF',
    borderRadius: 20,
    paddingHorizontal: 20,
    paddingVertical: 10,
    minWidth: 70,
  },
  sendButtonDisabled: {
    backgroundColor: '#B8B8B8',
  },
  sendButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  bottomNav: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    backgroundColor: '#fff',
    height: 60,
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: '#e1e1e1'
  },
  navItem: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center'
  },
  navIcon: {
    width: 28,
    height: 28,
    resizeMode: 'contain'
  },
  messageRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    marginVertical: 6,
  },
  userMessageRow: {
    flexDirection: 'row-reverse',
  },
  assistantMessageRow: {
    flexDirection: 'row',
  },
  profilePictureContainer: {
    width: 50,
    height: 50,
    marginHorizontal: 8,
  },
  profilePicture: {
    width: 50,
    height: 50,
    borderRadius: 25,
  },
  profilePicturePlaceholder: {
    backgroundColor: '#E8E8E8',
    justifyContent: 'center',
    alignItems: 'center',
  },
  profilePicturePlaceholderText: {
    fontSize: 14,
    color: '#666',
  },
  userMessageText: {
    color: '#FFFFFF',
  },
  assistantMessageText: {
    color: '#000000',
  },
  });

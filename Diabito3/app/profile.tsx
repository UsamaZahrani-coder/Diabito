import React, { useState, useEffect } from 'react';
import { StyleSheet, View, Text, TextInput, Image, TouchableOpacity, Alert, ScrollView, SafeAreaView, Platform, Switch } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface UserProfile {
  username: string;
  email: string;
  profile_picture: string | null;
  weight: string;
  height: string;
  age: string;
}

const API_URL = "http://192.168.0.111:5000";


export default function Profile() {
  const [profile, setProfile] = useState<UserProfile>({
    username: '',
    email: '',
    profile_picture: null,
    weight: '',
    height: '',
    age: ''
  });
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [notifications, setNotifications] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadProfile();
    loadSettings();
  }, []);

  const loadProfile = async () => {
    try {
      const token = await AsyncStorage.getItem('token');
      if (!token) {
        router.replace('/login');
        return;
      }

      const response = await axios.get(`${API_URL}/get-profile`, {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (response.data && typeof response.data === 'object') {
        console.log('Profile data received:', response.data);
        const userData = response.data as {
          username: string;
          email: string;
          profile_picture: string | null;
          weight: number;
          height: number;
          age: number;
        };
        
        setProfile({
          username: userData.username || '',
          email: userData.email || '',
          profile_picture: userData.profile_picture ? `${API_URL}/static/uploads/${userData.profile_picture}` : null,
          weight: userData.weight ? userData.weight.toString() : '',
          height: userData.height ? userData.height.toString() : '',
          age: userData.age ? userData.age.toString() : ''
        });
        setError(null);
      } else {
        console.error('No data received from get-profile endpoint');
      }
    } catch (error: any) {
      console.error('Error loading profile:', error);
      const errorMessage = error.response?.status === 404
        ? 'Profile not found. Please try logging in again.'
        : 'Failed to load profile. Please check your connection.';
      Alert.alert('Error', errorMessage);
      if (error.response?.status === 401 || error.response?.status === 404) {
        router.replace('/login');
      }
    }
  };

  const updateProfile = async () => {
    try {
      setLoading(true);
      const token = await AsyncStorage.getItem('token');
      if (!token) {
        router.replace('/login');
        return;
      }

      await axios.put(`${API_URL}/update-profile`, {
        weight: parseFloat(profile.weight) || 0,
        height: parseFloat(profile.height) || 0,
        age: parseInt(profile.age) || 0
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });

      Alert.alert('Success', 'Profile updated successfully');
      await loadProfile(); // Reload profile to ensure data consistency
    } catch (error) {
      console.error('Error updating profile:', error);
      Alert.alert('Error', 'Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (status !== 'granted') {
      Alert.alert('Permission needed', 'Please grant camera roll permissions to use this feature.');
      return;
    }
    
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8
    });
    
    if (!result.canceled && result.assets[0]) {
      try {
        setLoading(true);
        const token = await AsyncStorage.getItem('token');
        if (!token) {
          router.replace('/login');
          return;
        }

        const formData = new FormData();
        const uri = result.assets[0].uri;
        const fileExtension = uri.split('.').pop()?.toLowerCase() || 'jpeg';

        // Ensure valid image type
        if (!['jpg', 'jpeg', 'png'].includes(fileExtension)) {
          throw new Error('Invalid image format. Please use JPG, JPEG, or PNG.');
        }

        const imageFile = {
          uri: Platform.OS === 'ios' ? uri.replace('file://', '') : uri,
          type: `image/${fileExtension === 'jpg' ? 'jpeg' : fileExtension}`,
          name: `profile.${fileExtension}`
        };

        formData.append('profile_picture', imageFile as any);

        const response = await axios.post(`${API_URL}/profile-picture`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
            Authorization: `Bearer ${token}`
          }
        });

        if (response.data && typeof response.data === 'object' && 'profile_picture' in response.data) {
          // Update profile with the complete URL from the server
          const profilePictureUrl = `${API_URL}/static/uploads/${response.data.profile_picture}`;
          setProfile(prev => ({ ...prev, profile_picture: profilePictureUrl }));
          Alert.alert('Success', 'Profile picture updated successfully');
        } else {
          throw new Error('Invalid response format');
        }
      } catch (error) {
        console.error('Error uploading profile picture:', error);
        Alert.alert('Error', 'Failed to upload profile picture');
      } finally {
        setLoading(false);
      }
    }
  };

  const loadSettings = async () => {
    try {
      const darkModeSetting = await AsyncStorage.getItem('darkMode');
      const notificationsSetting = await AsyncStorage.getItem('notifications');

      setDarkMode(darkModeSetting === 'true');
      setNotifications(notificationsSetting !== 'false');
    } catch (error) {
      console.error('Error loading settings:', error);
    }
  };

  const handleLogout = async () => {
    try {
      await AsyncStorage.multiRemove(['token', 'username']);
      router.replace('/login');
    } catch (error) {
      console.error('Error logging out:', error);
      Alert.alert('Error', 'Failed to log out');
    }
  };

  const toggleDarkMode = async (value: boolean) => {
    setDarkMode(value);
    await AsyncStorage.setItem('darkMode', value.toString());
  };

  const toggleNotifications = async (value: boolean) => {
    setNotifications(value);
    await AsyncStorage.setItem('notifications', value.toString());
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={styles.scrollView}>
        <View style={styles.content}>
          <Text style={styles.headerTitle}>Profile</Text>

          <View style={styles.profileSection}>
            <TouchableOpacity onPress={pickImage} style={styles.profileImageContainer}>
              {profile.profile_picture ? (
                <Image
                  source={{ uri: profile.profile_picture }}
                  style={styles.profileImage}
                />
              ) : (
                <View style={styles.profileImagePlaceholder}>
                  <Text style={styles.profileImagePlaceholderText}>Add Photo</Text>
                </View>
              )}
            </TouchableOpacity>

            <Text style={styles.username}>{profile.username}</Text>
            <Text style={styles.email}>{profile.email}</Text>
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Personal Information</Text>
            <View style={styles.inputGroup}>
              <TextInput
                style={styles.input}
                placeholder="Weight (kg)"
                value={profile.weight}
                onChangeText={(text) => setProfile(prev => ({ ...prev, weight: text }))}
                keyboardType="numeric"
              />
              <TextInput
                style={styles.input}
                placeholder="Height (cm)"
                value={profile.height}
                onChangeText={(text) => setProfile(prev => ({ ...prev, height: text }))}
                keyboardType="numeric"
              />
              <TextInput
                style={styles.input}
                placeholder="Age"
                value={profile.age}
                onChangeText={(text) => setProfile(prev => ({ ...prev, age: text }))}
                keyboardType="numeric"
              />
            </View>
            <TouchableOpacity style={styles.updateButton} onPress={updateProfile}>
              <Text style={styles.updateButtonText}>Update Profile</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>App Settings</Text>
            <View style={styles.settingItem}>
              <Text style={styles.settingLabel}>Dark Mode</Text>
              <Switch
                value={darkMode}
                onValueChange={toggleDarkMode}
                trackColor={{ false: '#767577', true: '#81b0ff' }}
                thumbColor={darkMode ? '#f5dd4b' : '#f4f3f4'}
              />
            </View>
            <View style={styles.settingItem}>
              <Text style={styles.settingLabel}>Notifications</Text>
              <Switch
                value={notifications}
                onValueChange={toggleNotifications}
                trackColor={{ false: '#767577', true: '#81b0ff' }}
                thumbColor={notifications ? '#f5dd4b' : '#f4f3f4'}
              />
            </View>
          </View>

          <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
            <Text style={styles.logoutButtonText}>Logout</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>

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
            source={require('../icons/chat.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => router.push('/profile')}
        >
          <Image 
            source={require('../icons/profile_cliked.png')}
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
  scrollView: {
    flex: 1
  },
  content: {
    padding: 20
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 20
  },
  profileSection: {
    alignItems: 'center',
    marginBottom: 30
  },
  profileImageContainer: {
    width: 120,
    height: 120,
    borderRadius: 60,
    marginBottom: 15,
    overflow: 'hidden',
    backgroundColor: '#e1e1e1'
  },
  profileImage: {
    width: '100%',
    height: '100%'
  },
  profileImagePlaceholder: {
    width: '100%',
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center'
  },
  profileImagePlaceholderText: {
    color: '#666',
    fontSize: 16
  },
  username: {
    fontSize: 24,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 5
  },
  email: {
    fontSize: 16,
    color: '#7f8c8d'
  },
  section: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 15
  },
  inputGroup: {
    marginBottom: 15
  },
  input: {
    height: 50,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    paddingHorizontal: 15,
    marginBottom: 10,
    fontSize: 16
  },
  updateButton: {
    backgroundColor: '#000000',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center'
  },
  updateButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600'
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0'
  },
  settingLabel: {
    fontSize: 16,
    color: '#2c3e50'
  },
  unitsToggle: {
    flexDirection: 'row',
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    padding: 2
  },
  unitButton: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 6
  },
  unitButtonActive: {
    backgroundColor: '#fff',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 2
  },
  unitButtonText: {
    fontSize: 14,
    color: '#2c3e50'
  },
  unitButtonTextActive: {
    color: '#007AFF',
    fontWeight: '600'
  },
  logoutButton: {
    backgroundColor: '#ff3b30',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 20
  },
  logoutButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600'
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
  }
});
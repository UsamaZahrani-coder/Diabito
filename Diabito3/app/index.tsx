import React, { useState, useEffect } from 'react';
import { StyleSheet, View, Text, TextInput, Button, Image, ActivityIndicator, Alert, ScrollView, TouchableOpacity, SafeAreaView } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import type { AxiosError } from 'axios';
import { router } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface ApiResponse {
  carbs: number;
  sugar: number;
  protein: number;
  correction_dose: number;
  carb_dose: number;
  bolus_dose: number;
  bolus_dose_ceil: number;
}

const API_URL = "http://192.168.0.111:5000";

const recommendations = [
  {
    title: 'Target Range',
    text: 'Keep your glucose between 80-130 mg/dL before meals'
  },
  {
    title: 'Meal Timing',
    text: 'Try to eat at regular intervals to maintain stable glucose levels'
  },
  {
    title: 'Exercise Tip',
    text: '30 minutes of daily exercise can help improve insulin sensitivity'
  },
  {
    title: 'Hydration',
    text: 'Drink plenty of water throughout the day to help regulate blood sugar'
  },
  {
    title: 'Sleep Schedule',
    text: 'Aim for 7-8 hours of sleep to help maintain healthy glucose levels'
  },
  {
    title: 'Stress Management',
    text: 'Practice relaxation techniques to prevent stress-induced glucose spikes'
  },
  {
    title: 'Meal Planning',
    text: 'Plan your meals ahead to ensure balanced nutrition and consistent carb intake'
  },
  {
    title: 'Blood Sugar Monitoring',
    text: 'Check your blood sugar 2-3 hours after meals to understand food effects'
  },
  {
    title: 'Foot Care',
    text: 'Inspect your feet daily for cuts, blisters, or signs of infection'
  },
  {
    title: 'Medication Schedule',
    text: 'Take medications at the same time each day for better glucose control'
  },
  {
    title: 'Carb Counting',
    text: 'Learn to accurately count carbohydrates to better manage insulin doses'
  },
  {
    title: 'Regular Check-ups',
    text: 'Schedule regular visits with your healthcare team for comprehensive care'
  },
  {
    title: 'Emergency Kit',
    text: 'Keep a diabetes emergency kit with glucose tablets and supplies'
  },
  {
    title: 'Physical Activity',
    text: 'Include both aerobic exercises and strength training in your routine'
  },
  {
    title: 'Portion Control',
    text: 'Use measuring tools and visual guides to control portion sizes'
  },
  {
    title: 'Alcohol Awareness',
    text: 'If drinking alcohol, always eat food to prevent low blood sugar'
  },
  {
    title: 'Travel Tips',
    text: 'Pack extra diabetes supplies and keep them in carry-on luggage'
  },
  {
    title: 'Sick Day Rules',
    text: 'Monitor blood sugar more frequently when ill and stay hydrated'
  },
  {
    title: 'Healthy Snacking',
    text: 'Keep low-carb snacks handy for between meal blood sugar control'
  },
  {
    title: 'Weather Impact',
    text: 'Be aware that hot or cold weather can affect blood glucose levels'
  },
  {
    title: 'Fiber Intake',
    text: 'Include high-fiber foods to help slow down glucose absorption'
  },
  {
    title: 'Medication Storage',
    text: 'Store insulin and other medications properly to maintain their effectiveness'
  },
  {
    title: 'Blood Pressure',
    text: 'Monitor your blood pressure regularly as it affects diabetes management'
  },
  {
    title: 'Glycemic Index',
    text: 'Choose foods with low glycemic index to prevent blood sugar spikes'
  },
  {
    title: 'Exercise Timing',
    text: 'Exercise at consistent times to better predict blood sugar responses'
  },
  {
    title: 'Dental Health',
    text: 'Maintain good oral hygiene as diabetes can increase risk of gum disease'
  },
  {
    title: 'Skin Care',
    text: 'Keep skin clean and moisturized to prevent diabetes-related complications'
  },
  {
    title: 'Stress Signs',
    text: 'Learn to recognize stress signals that may affect your glucose levels'
  },
  {
    title: 'Social Support',
    text: 'Connect with diabetes support groups for shared experiences and tips'
  },
  {
    title: 'Eye Care',
    text: 'Get regular eye exams as diabetes can affect vision health'
  },
  {
    title: 'Meal Spacing',
    text: 'Space meals 4-5 hours apart to maintain steady blood sugar levels'
  },
  {
    title: 'Sleep Quality',
    text: 'Create a bedtime routine to improve sleep quality and glucose control'
  },
  {
    title: 'Exercise Recovery',
    text: 'Monitor blood sugar during and after exercise for proper management'
  },
  {
    title: 'Mindful Eating',
    text: 'Practice mindful eating to better control portions and food choices'
  },
  {
    title: 'Medication Timing',
    text: 'Time medications with meals according to your doctor\'s instructions'
  },
  {
    title: 'Travel Planning',
    text: 'Research medical facilities at your destination when planning trips'
  },
  {
    title: 'Seasonal Care',
    text: 'Adjust diabetes management for different seasons and weather changes'
  },
  {
    title: 'Emotional Health',
    text: 'Monitor your emotional well-being as it affects diabetes management'
  },
  {
    title: 'Activity Tracking',
    text: 'Keep a log of physical activities to understand their impact on glucose'
  },
  {
    title: 'Emergency Planning',
    text: 'Have a plan for managing diabetes during unexpected situations'
  },
  {
    title: 'Nutrition Labels',
    text: 'Read nutrition labels carefully to make informed food choices'
  },
  {
    title: 'Label Reading',
    text: 'Check nutrition labels for hidden sugars and serving sizes'
  },
  {
    title: 'Exercise Timing',
    text: 'Time workouts to avoid low blood sugar, typically 1-2 hours after meals'
  },
  {
    title: 'Mindful Eating',
    text: 'Eat slowly and mindfully to better recognize hunger and fullness cues'
  },
  {
    title: 'Support System',
    text: 'Build a support network of family and friends who understand diabetes'
  },
  {
    title: 'Dental Care',
    text: 'Maintain good oral hygiene as diabetes can affect gum health'
  },
  {
    title: 'Stress Signals',
    text: 'Learn to recognize how stress affects your blood sugar levels'
  },
  {
    title: 'Seasonal Care',
    text: 'Adjust your diabetes management plan for different seasons and weather conditions'
  },
  {
    title: 'Social Dining',
    text: 'Plan ahead for restaurant meals by reviewing menus and counting carbs beforehand'
  },
  {
    title: 'Mental Health',
    text: 'Regular mental health check-ins are important as diabetes can affect emotional well-being'
  },
  {
    title: 'Advanced Exercise',
    text: 'Mix high-intensity intervals with moderate activities for better glucose control'
  },
  {
    title: 'Morning Routine',
    text: 'Establish a consistent morning routine to help stabilize blood sugar levels'
  },
  {
    title: 'Night Management',
    text: 'Check blood sugar before bed and keep glucose tablets nearby while sleeping'
  },
  {
    title: 'Travel Planning',
    text: 'Research medical facilities at your destination when planning trips'
  },
  {
    title: 'Work Management',
    text: 'Keep diabetes supplies organized and accessible at your workplace'
  },
  {
    title: 'Technology Use',
    text: 'Utilize diabetes management apps and tools to track your health data'
  },
  {
    title: 'Emergency Contacts',
    text: 'Keep emergency contacts updated and informed about your diabetes management'
  },
  {
    title: 'Seasonal Foods',
    text: 'Learn about seasonal foods\'s impact on blood sugar and adjust meals accordingly'
  },
  {
    title: 'Exercise Recovery',
    text: 'Monitor blood sugar during post-workout recovery and adjust insulin as needed'
  },
  {
    title: 'Stress Relief',
    text: 'Develop healthy stress-relief techniques that don\'t impact blood sugar levels'
  },
  {
    title: 'Social Support',
    text: 'Join diabetes support groups to share experiences and learn from others'
  },
  {
    title: 'Medical Records',
    text: 'Keep detailed records of your medical history and diabetes management plan'
  },
  {
    title: 'Alternative Exercise',
    text: 'Explore low-impact exercises like swimming or yoga for variety in your routine'
  },
  {
    title: 'Diet Flexibility',
    text: 'Learn to adapt your meal plan while maintaining good blood sugar control'
  },
  {
    title: 'Sleep Hygiene',
    text: 'Practice good sleep habits to help maintain stable blood sugar levels'
  },
  {
    title: 'Medication Timing',
    text: 'Coordinate medication timing with meals and daily activities for better control'
  },
  {
    title: 'Self-Advocacy',
    text: 'Learn to effectively communicate your needs to healthcare providers'
  }
];

export default function App() {
  const [activeSection, setActiveSection] = useState('dashboard');
  const [currentRecommendationIndex, setCurrentRecommendationIndex] = useState(0);
  const [username, setUsername] = useState('');
  const [profilePicture, setProfilePicture] = useState<string | null>(null);
  const [visibleRecommendations, setVisibleRecommendations] = useState<typeof recommendations>([]);

  useEffect(() => {
    // Load username from AsyncStorage
    const loadUserData = async () => {
      try {
        const token = await AsyncStorage.getItem('token');
        if (token) {
          const response = await axios.get(`${API_URL}/get-profile`, {
            headers: { Authorization: `Bearer ${token}` }
          });
          
          if (response.data && typeof response.data === 'object') {
            setUsername(response.data.username || '');
            setProfilePicture(response.data.profile_picture ? 
              `${API_URL}/static/uploads/${response.data.profile_picture}` : null);
          }
        }
      } catch (error) {
        console.error('Error loading user data:', error);
      }
    };
    loadUserData();
    // Initialize with first 3 recommendations
    setVisibleRecommendations(recommendations.slice(0, 3));

    const timer = setInterval(() => {
      setVisibleRecommendations(currentVisible => {
        const currentIndex = recommendations.findIndex(r => r === currentVisible[0]);
        const nextStartIndex = (currentIndex + 1) % recommendations.length;
        return [
          ...recommendations.slice(nextStartIndex, Math.min(nextStartIndex + 3, recommendations.length)),
          ...recommendations.slice(0, Math.max(0, 3 - (recommendations.length - nextStartIndex)))
        ];
      });
    }, 60000); // Changed to 1 minute (60000 milliseconds) for more frequent updates

    return () => clearInterval(timer);
  }, []);


  const navigateToSection = (section: string) => {
    if (section === 'calculator') {
      router.push('/calculator');
    } else {
      setActiveSection(section);
    }
  };

  const renderDashboard = () => (
    <View style={styles.dashboardContainer}>
      <View style={styles.welcomeHeader}>
        <View style={styles.userInfo}>
          <Text style={styles.hiText}>Hi,</Text>
          <Text style={styles.usernameText}>{username}</Text>
        </View>
        <View style={styles.profileContainer}>
          {profilePicture ? (
            <Image
              source={{ uri: profilePicture }}
              style={styles.profileImage}
            />
          ) : (
            <View style={styles.profileImagePlaceholder}>
              <Text style={styles.profileImagePlaceholderText}>?</Text>
            </View>
          )}
        </View>
      </View>
      
      <View style={styles.quickActions}>
        <TouchableOpacity 
          style={styles.actionCard}
          onPress={() => navigateToSection('calculator')}
        >
          <Text style={styles.actionTitle}>Calculate Insulin</Text>
          <Text style={styles.actionDescription}>Calculate your insulin dose based on food and glucose levels</Text>
        </TouchableOpacity>

        <TouchableOpacity 
          style={styles.actionCard}
          onPress={() => router.push('/profile')}
        >
          <Text style={styles.actionTitle}>Profile</Text>
          <Text style={styles.actionDescription}>View and update your profile settings</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.recommendationsSection}>
        <Text style={styles.sectionTitle}>Daily Recommendations</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.recommendationsScroll}>
          {visibleRecommendations.map((recommendation, index) => (
            <View key={index} style={styles.recommendationCard}>
              <Text style={styles.recommendationTitle}>{recommendation.title}</Text>
              <Text style={styles.recommendationText}>{recommendation.text}</Text>
            </View>
          ))}
        </ScrollView>
      </View>

      <View style={styles.tipsSection}>
        <Text style={styles.sectionTitle}>Health Tips</Text>
        <View style={styles.tipCard}>
          <Text style={styles.tipTitle}>Tip of the Day</Text>
          <Text style={styles.tipText}>Remember to check your blood sugar before and after physical activity</Text>
        </View>
      </View>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView>
        {renderDashboard()}
      </ScrollView>
      <View style={styles.bottomNav}>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => navigateToSection('dashboard')}
        >
          <Image 
            source={activeSection === 'dashboard' ? require('../icons/home clicked.png') : require('../icons/home.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => navigateToSection('calculator')}
        >
          <Image 
            source={activeSection === 'calculator' ? require('../icons/calculator_clicked.png') : require('../icons/calculator.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => router.push('/chat')}
        >
          <Image 
            source={activeSection === 'chat' ? require('../icons/chat_cliked.png') : require('../icons/chat.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.navItem}
          onPress={() => router.push('/profile')}
        >
          <Image 
            source={activeSection === 'profile' ? require('../icons/profile_cliked.png') : require('../icons/Profile.png')}
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
  dashboardContainer: {
    padding: 20
  },
  welcomeHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 20,
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  profileContainer: {
    marginLeft: 15
  },
  profileImage: {
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 2,
    borderColor: '#fff'
  },
  profileImagePlaceholder: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#e1e1e1',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#fff'
  },
  profileImagePlaceholderText: {
    fontSize: 20,
    color: '#666'
  },
  userInfo: {
    flex: 1
  },
  hiText: {
    fontSize: 16,
    color: '#666',
    marginBottom: 2
  },
  usernameText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c3e50'
  },
  quickActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20
  },
  actionCard: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 12,
    width: '48%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  actionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 8
  },
  actionDescription: {
    fontSize: 12,
    color: '#7f8c8d'
  },
  recommendationsSection: {
    marginBottom: 20
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 15
  },
  recommendationsScroll: {
    marginBottom: 10
  },
  recommendationCard: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 12,
    marginRight: 15,
    width: 250,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    marginBottom: 10
  },
  recommendationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 8
  },
  recommendationText: {
    fontSize: 14,
    color: '#7f8c8d'
  },
  tipsSection: {
    marginBottom: 20
  },
  tipCard: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  tipTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 8
  },
  tipText: {
    fontSize: 14,
    color: '#7f8c8d'
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
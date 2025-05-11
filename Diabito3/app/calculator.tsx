import React, { useState, useEffect } from 'react';
import { StyleSheet, View, Text, TextInput, Image, ActivityIndicator, Alert, ScrollView, SafeAreaView, Platform, TouchableOpacity } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { router } from 'expo-router';
import axios from 'axios';
import type { AxiosError } from 'axios';
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

interface CalculationHistoryItem extends ApiResponse {
  id?: number;
  timestamp: string;
  imageUri?: string;
  image_filename?: string;
  glucose_value: string;
  target_glucose: string;
  insulin_sensitivity: string;
  carb_insulin_ratio: string;
}

const API_URL = "http://192.168.0.111:5000";

export default function Calculator() {
  const [glucose, setGlucose] = useState('');
  const [targetGlucose, setTargetGlucose] = useState('');
  const [insulinSensitivity, setInsulinSensitivity] = useState('');
  const [carbInsulinRatio, setCarbInsulinRatio] = useState('');
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [history, setHistory] = useState<CalculationHistoryItem[]>([]);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const token = await AsyncStorage.getItem('token');
      if (!token) {
        console.log('No token found, skipping history load');
        return;
      }

      const response = await axios.get(`${API_URL}/calculation-history`, {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (response.data && Array.isArray(response.data)) {
        // Transform backend data to match frontend format
        const historyItems = response.data.map((item: any) => ({
          ...item,
          imageUri: item.image_filename ? `${API_URL}/static/uploads/${item.image_filename}` : ''
        }));
        setHistory(historyItems);
      }
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  // No need to save to history manually as the backend will handle it
  const saveToHistory = async (calculationResult: ApiResponse) => {
    // Refresh history from server after calculation
    await loadHistory();
  };

  const getFileSizeInBytes = (base64String: string): number => {
    const padding = base64String.endsWith('==') ? 2 : base64String.endsWith('=') ? 1 : 0;
    return (base64String.length * 0.75) - padding;
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
      aspect: [4, 3],
      quality: 0.8,
      base64: true,
      exif: true
    });
    
    if (!result.canceled && result.assets[0]) {
      const selectedImage = result.assets[0];
      let compressionQuality = 0.8;
      
      if (selectedImage.base64) {
        const maxSize = 300 * 1024; // 300KB in bytes
        const initialSize = getFileSizeInBytes(selectedImage.base64);
        
        if (initialSize > maxSize) {
          compressionQuality = Math.max(0.1, Math.min(0.8, (maxSize / initialSize) * 0.8));
          
          const recompressedResult = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [4, 3],
            quality: compressionQuality,
            base64: true
          });
          
          if (!recompressedResult.canceled && recompressedResult.assets[0]) {
            const finalSize = recompressedResult.assets[0].base64 
              ? getFileSizeInBytes(recompressedResult.assets[0].base64)
              : initialSize;
              
            if (finalSize <= maxSize || compressionQuality <= 0.1) {
              setImage(recompressedResult.assets[0].uri);
              return;
            }
          }
        }
      }
      
      setImage(selectedImage.uri);
    }
  };

  const calculateInsulin = async () => {
    if (!glucose || !targetGlucose || !insulinSensitivity || !carbInsulinRatio || !image) {
      Alert.alert('Error', 'Please fill all fields and upload an image');
      return;
    }

    // Validate image format
    const imageExtension = image.split('.').pop()?.toLowerCase();
    const validExtensions = ['jpg', 'jpeg', 'png'];
    if (!imageExtension || !validExtensions.includes(imageExtension)) {
      Alert.alert('Error', 'Please upload a valid image (JPG, JPEG, or PNG)');
      return;
    }
    
    setLoading(true);
    const formData = new FormData();
    formData.append('glucose_value', glucose);
    formData.append('target_glucose', targetGlucose);
    formData.append('insulin_sensitivity', insulinSensitivity);
    formData.append('carb_insulin_ratio', carbInsulinRatio);
    
    const imageUri = Platform.OS === 'ios' ? image.replace('file://', '') : image;
    const imageFileName = imageUri.split('/').pop() || 'food.jpg';
    
    try {
      formData.append('image', {
        uri: imageUri,
        type: `image/${imageExtension === 'jpg' ? 'jpeg' : imageExtension}`,
        name: imageFileName
      } as any);

      // Get token for authenticated requests
      const token = await AsyncStorage.getItem('token');
      
      const response = await axios.post<ApiResponse>(`${API_URL}/`, formData, {
        headers: { 
          'Content-Type': 'multipart/form-data',
          'Accept': 'application/json',
          'Authorization': token ? `Bearer ${token}` : ''
        },
        timeout: 30000
      });
      
      if (response.data) {
        setResult(response.data);
        await saveToHistory(response.data);
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (error) {
      const axiosError = error as AxiosError;
      let errorMessage = 'Failed to process the image. Please try again.';
      
      if (axiosError.code === 'ECONNABORTED') {
        errorMessage = 'Request timed out. Please check your internet connection and try again.';
      } else if (axiosError.response?.status === 413) {
        errorMessage = 'Image file is too large. Please choose a smaller image or compress the current one.';
      } else if (axiosError.response?.status === 415) {
        errorMessage = 'Unsupported image format. Please use JPG, JPEG, or PNG.';
      } else if (axiosError.response?.data && typeof axiosError.response.data === 'object' && 'error' in axiosError.response.data) {
        errorMessage = (axiosError.response.data as { error: string }).error;
      }
      
      Alert.alert('Error', errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView style={{ flex: 1 }}>
        <View style={styles.content}>
          <Text style={styles.headerTitle}>Insulin Calculator</Text>
          <TouchableOpacity 
            style={styles.manualButton}
            onPress={() => router.push('/manual-calculator')}
          >
            <Text style={styles.manualButtonText}>Switch to Manual Calculator</Text>
          </TouchableOpacity>
          <View style={styles.form}>
            <TextInput
              style={styles.input}
              placeholder="Current Glucose Level"
              value={glucose}
              onChangeText={setGlucose}
              keyboardType="numeric"
            />
            <TextInput
              style={styles.input}
              placeholder="Target Glucose Level"
              value={targetGlucose}
              onChangeText={setTargetGlucose}
              keyboardType="numeric"
            />
            <TextInput
              style={styles.input}
              placeholder="Insulin Sensitivity Factor"
              value={insulinSensitivity}
              onChangeText={setInsulinSensitivity}
              keyboardType="numeric"
            />
            <TextInput
              style={styles.input}
              placeholder="Carb to Insulin Ratio"
              value={carbInsulinRatio}
              onChangeText={setCarbInsulinRatio}
              keyboardType="numeric"
            />
            
            <View style={styles.imageSection}>
              <View style={styles.imageButtons}>
                <TouchableOpacity onPress={pickImage} style={[styles.uploadButton, styles.imageButton]}>
                  <Image
                    source={require('../icons/Upload_food_image.png')}
                    style={styles.uploadIcon}
                  />
                  <Text style={styles.uploadText}>Upload Image</Text>
                </TouchableOpacity>
                <TouchableOpacity 
                  onPress={async () => {
                    const { status } = await ImagePicker.requestCameraPermissionsAsync();
                    if (status !== 'granted') {
                      Alert.alert('Permission needed', 'Please grant camera permissions to use this feature.');
                      return;
                    }
                    const result = await ImagePicker.launchCameraAsync({
                      mediaTypes: ImagePicker.MediaTypeOptions.Images,
                      allowsEditing: true,
                      aspect: [4, 3],
                      quality: 0.8,
                      base64: true
                    });
                    if (!result.canceled && result.assets[0]) {
                      setImage(result.assets[0].uri);
                    }
                  }} 
                  style={[styles.uploadButton, styles.imageButton]}
                >
                  <Image
                    source={require('../icons/Open_Camera.png')}
                    style={styles.uploadIcon}
                  />
                  <Text style={styles.uploadText}>Open Camera</Text>
                </TouchableOpacity>
              </View>
              {image && <Image source={{ uri: image }} style={styles.imagePreview} />}
            </View>
            
            <View style={styles.calculateButtonContainer}>
              <TouchableOpacity style={styles.calculateButton} onPress={calculateInsulin}>
                <Text style={styles.calculateButtonText}>Calculate Insulin</Text>
              </TouchableOpacity>
            </View>
            
            {loading && <ActivityIndicator size="large" color="#007AFF" style={styles.loader} />}
            
            {result && (
              <View style={styles.resultContainer}>
                <Text style={styles.resultTitle}>Nutritional Information</Text>
                <View style={styles.resultSection}>
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>Carbohydrates:</Text>
                    <Text style={styles.resultValue}>{result.carbs?.toFixed(1)}g</Text>
                  </View>
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>Sugar:</Text>
                    <Text style={styles.resultValue}>{result.sugar?.toFixed(1)}g</Text>
                  </View>
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>Protein:</Text>
                    <Text style={styles.resultValue}>{result.protein?.toFixed(1)}g</Text>
                  </View>
                </View>

                <Text style={[styles.resultTitle, styles.secondTitle]}>Insulin Doses</Text>
                <View style={styles.resultSection}>
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>Correction Dose:</Text>
                    <Text style={styles.resultValue}>{result.correction_dose} units</Text>
                  </View>
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>Carb Dose:</Text>
                    <Text style={styles.resultValue}>{result.carb_dose} units</Text>
                  </View>
                  <View style={styles.resultRow}>
                    <Text style={styles.resultLabel}>Total Bolus Dose:</Text>
                    <Text style={styles.resultValue}>{result.bolus_dose} units</Text>
                  </View>
                  <View style={[styles.resultRow, styles.finalResult]}>
                    <Text style={[styles.resultLabel, styles.boldText]}>Rounded Bolus Dose:</Text>
                    <Text style={[styles.resultValue, styles.boldText]}>{result.bolus_dose_ceil} units</Text>
                  </View>
                </View>
              </View>
            )}

            {history.length > 0 && (
              <View style={styles.historyContainer}>
                <Text style={styles.historyTitle}>Calculation History</Text>
                {history.map((item, index) => (
                  <View key={index} style={styles.historyItem}>
                    <View style={styles.historyHeader}>
                      <Text style={styles.historyTimestamp}>
                        {new Date(item.timestamp).toLocaleString()}
                      </Text>
                    </View>
                    {item.imageUri && (
                      <Image 
                        source={{ uri: item.imageUri }} 
                        style={styles.historyImage}
                      />
                    )}
                    <View style={styles.historyDetails}>
                      <View style={styles.historyRow}>
                        <Text style={styles.historyLabel}>Glucose:</Text>
                        <Text style={styles.historyValue}>{item.glucose_value} mg/dL</Text>
                      </View>
                      <View style={styles.historyRow}>
                        <Text style={styles.historyLabel}>Target:</Text>
                        <Text style={styles.historyValue}>{item.target_glucose} mg/dL</Text>
                      </View>
                      <View style={styles.historyRow}>
                        <Text style={styles.historyLabel}>Carbs:</Text>
                        <Text style={styles.historyValue}>{item.carbs?.toFixed(1)}g</Text>
                      </View>
                      <View style={styles.historyRow}>
                        <Text style={styles.historyLabel}>Total Insulin:</Text>
                        <Text style={styles.historyValue}>{item.bolus_dose_ceil} units</Text>
                      </View>
                    </View>
                  </View>
                ))}
              </View>
            )}
          </View>
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
            source={require('../icons/calculator_clicked.png')}
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
            source={require('../icons/Profile.png')}
            style={styles.navIcon}
          />
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
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
  manualButton: {
    backgroundColor: '#fff',
    padding: 12,
    borderRadius: 8,
    marginBottom: 15,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: '#000'
  },
  manualButtonText: {
    color: '#000',
    fontSize: 17,
    fontWeight: '500'
  },
  container: {
    flex: 1,
    backgroundColor: '#f5f6fa'
  },
  content: {
    padding: 20
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 20,
    textAlign: 'center'
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 20,
    textAlign: 'left'
  },
  form: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3
  },
  input: {
    height: 50,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    paddingHorizontal: 15,
    marginBottom: 15,
    fontSize: 16
  },
  imageSection: {
    marginVertical: 15,
    alignItems: 'center'
  },
  imageButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginBottom: 15
  },
  uploadButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f0f0f0',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ddd',
    borderStyle: 'dashed'
  },
  imageButton: {
    flex: 0.48
  },
  uploadIcon: {
    width: 24,
    height: 24,
    marginRight: 8,
    resizeMode: 'contain'
  },
  uploadText: {
    fontSize: 16,
    color: '#666'
  },
  imagePreview: {
    width: 200,
    height: 150,
    marginTop: 10,
    borderRadius: 8
  },
  calculateButtonContainer: {
    marginVertical: 20
  },
  calculateButton: {
    backgroundColor: '#000',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: 8,
    width: '100%',
    alignItems: 'center'
  },
  calculateButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600'
  },
  loader: {
    marginVertical: 20
  },
  resultContainer: {
    marginTop: 20,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#2c3e50',
    marginBottom: 15
  },
  secondTitle: {
    marginTop: 20
  },
  resultSection: {
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 12
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee'
  },
  resultLabel: {
    fontSize: 14,
    color: '#2c3e50'
  },
  resultValue: {
    fontSize: 14,
    color: '#2c3e50',
    fontWeight: '500'
  },
  finalResult: {
    marginTop: 8,
    borderBottomWidth: 0,
    backgroundColor: '#e3f2fd',
    borderRadius: 6,
    padding: 8
  },
  boldText: {
    fontWeight: 'bold',
    color: '#1976d2'
  },
  historyContainer: {
    marginTop: 20,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    marginBottom: 20
  },
  historyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 15
  },
  historyItem: {
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    marginBottom: 15,
    padding: 12
  },
  historyHeader: {
    marginBottom: 10
  },
  historyTimestamp: {
    color: '#666',
    fontSize: 14
  },
  historyImage: {
    width: '100%',
    height: 150,
    borderRadius: 8,
    marginBottom: 10
  },
  historyDetails: {
    backgroundColor: '#fff',
    borderRadius: 6,
    padding: 10
  },
  historyRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 5
  },
  historyLabel: {
    fontSize: 14,
    color: '#2c3e50',
    fontWeight: '500'
  },
  historyValue: {
    fontSize: 14,
    color: '#2c3e50'
  },
  historyContainerStyle: {
    marginTop: 20,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    marginBottom: 20
  },
  historyTitleStyle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 15
  },
  historyItemStyle: {
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 12,
    marginBottom: 10
  },
  historyHeaderStyle: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8
  },
  historyTimestampText: {
    fontSize: 12,
    color: '#666',
    fontStyle: 'italic'
  },
  historyImageStyle: {
    width: '100%',
    height: 150,
    borderRadius: 8,
    marginBottom: 10
  },
  historyDetailsStyle: {
    backgroundColor: '#fff',
    borderRadius: 6,
    padding: 10
  },
  historyRowStyle: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 4
  },
  historyLabelText: {
    fontSize: 14,
    color: '#2c3e50',
    fontWeight: '500'
  },
  historyValueText: {
    fontSize: 14,
    color: '#2c3e50'
  }
});
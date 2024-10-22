import React, {useEffect, useState} from 'react';
import {
  View,
  Button,
  Text,
  StyleSheet,
  Platform,
  PermissionsAndroid,
  ActivityIndicator,
  Animated,
  TouchableOpacity,
} from 'react-native';
import Sound from 'react-native-sound';
import AudioRecorderPlayer from 'react-native-audio-recorder-player';
import RNFS from 'react-native-fs';
import notifee, {AndroidImportance} from '@notifee/react-native';
import Icon from 'react-native-vector-icons/FontAwesome';

const audioRecorderPlayer = new AudioRecorderPlayer();

function App(): React.JSX.Element {
  const [recording, setRecording] = useState<boolean>(false);
  const [recordingPath, setRecordingPath] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const requestPermissions = async (): Promise<void> => {
    if (Platform.OS === 'android') {
      try {
        const audioPermission = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
        );
        if (audioPermission !== PermissionsAndroid.RESULTS.GRANTED) {
          console.warn('Audio permission denied');
          return;
        }

        const notificationPermission = await notifee.requestPermission();
        if (!notificationPermission) {
          console.warn('Notification permission denied');
          return;
        }
      } catch (err) {
        console.warn(err);
      }
    }
  };

  const startRecording = async (): Promise<void> => {
    try {
      setRecording(true);
      const path = `${RNFS.DocumentDirectoryPath}/${Date.now()}.m4a`;
      const result = await audioRecorderPlayer.startRecorder(path);
      setRecordingPath(result);
      audioRecorderPlayer.addRecordBackListener(e => {
        console.log('Recording...', e.currentPosition);
      });
    } catch (error) {
      console.warn('Error starting recording:', error);
    }
  };

  const stopRecording = async (): Promise<void> => {
    try {
      const result = await audioRecorderPlayer.stopRecorder();
      setRecording(false);
      audioRecorderPlayer.removeRecordBackListener();
      console.log('Recording stopped:', result);
    } catch (error) {
      console.warn('Error stopping recording:', error);
    }
  };

  const playRecording = async (): Promise<void> => {
    if (!recordingPath) return;

    try {
      setIsPlaying(true);
      await audioRecorderPlayer.startPlayer(recordingPath);
      audioRecorderPlayer.addPlayBackListener(async e => {
        if (e.currentPosition + 30 >= e.duration) {
          stopPlaying();
        }
      });
    } catch (error) {
      console.warn('Error playing recording:', error);
    }
  };

  const stopPlaying = async (): Promise<void> => {
    try {
      await audioRecorderPlayer.stopPlayer();
      console.log('Audio stopped');
      setIsPlaying(false);
      audioRecorderPlayer.removePlayBackListener();
    } catch (error) {
      console.warn('Error stopping playback:', error);
    }
  };

  const uploadAudioToServer = async (uri: string) => {
    const fileExists = await RNFS.exists(uri);
    if (!fileExists) {
      console.error('File does not exist at the specified path:', uri);
      return;
    }

    const formData = new FormData();
    const fileName = uri.substring(uri.lastIndexOf('/') + 1);

    formData.append('file', {
      uri: uri,
      type: 'audio/m4a', // Correct MIME type for m4a
      name: fileName,
    });

    try {
      const response = await fetch('http://localhost:8801/uploadfile/', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      fetchAudio();
      return data;
    } catch (error) {
      console.error('Error uploading audio file:', error);
    }
  };

  const getLikelihoodColor = (percentage: number) => {
    if (percentage < 50) return 'green';
    if (percentage < 75) return 'yellow';
    return 'red';
  };

  const classifyRecording = async (): Promise<void> => {
    if (!recordingPath) {
      console.warn('No recording available to classify.');
      return;
    }

    setIsLoading(true);
    setPredictionResult(null);

    const channelId = await notifee.createChannel({
      id: 'default',
      name: 'Default Channel',
    });

    const result = await uploadAudioToServer(recordingPath);
    console.log('Model output: ' + result['prediction']);

    setPredictionResult(result['prediction']);
    setIsLoading(false);

    if (result['prediction'] >= 0.5) {
      await notifee.displayNotification({
        title: 'Health Checkup',
        body: "Hi - we noticed you've been showing subtle signs of mental decline recently and we think it's time for a checkup.",
        android: {
          channelId: channelId,
          importance: AndroidImportance.HIGH,
        },
      });
    }
  };

  const fetchAudio = async () => {
    const audioUrl = 'http://localhost:8801/get-audio';
    const sound = new Sound(audioUrl, Sound.MAIN_BUNDLE, error => {
      if (error) {
        console.error('Failed to load sound', error);
        return;
      }
      sound.play(success => {
        if (!success) {
          console.log('Sound did not play');
        }
      });
    });
  };

  useEffect(() => {
    requestPermissions();
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.appName}>ALI</Text>
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[
            styles.button,
            {backgroundColor: recording ? 'red' : '#4591ed'},
          ]}
          onPress={recording ? stopRecording : startRecording}
          disabled={isLoading}>
          <Icon
            name={recording ? 'stop' : 'microphone'}
            size={30}
            color="white"
          />
        </TouchableOpacity>
      </View>

      {recordingPath ? (
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[
              styles.button,
              {backgroundColor: isPlaying ? 'red' : 'green'}, // Change the color when playing
            ]}
            onPress={isPlaying ? stopPlaying : playRecording}
            disabled={recording}>
            <Icon name={isPlaying ? 'stop' : 'play'} size={30} color="white" />
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.button}
            onPress={classifyRecording}
            disabled={recording}>
            <Icon name="send" size={30} color="white" />
          </TouchableOpacity>
        </View>
      ) : null}

      {isLoading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="blue" />
        </View>
      ) : predictionResult !== null ? (
        <Text
          style={[
            styles.resultText,
            {color: getLikelihoodColor(predictionResult * 100)},
          ]}>
          Likelihood of Alzheimer's: {(predictionResult * 100).toFixed(0)}%
        </Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#b6e3f2',
  },
  appName: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 20,
    color: '#1e3c72',
  },
  recordingText: {
    fontSize: 18,
    color: 'red',
    marginVertical: 20,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    marginTop: 20,
  },
  button: {
    flex: 1,
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: 5,
    marginHorizontal: 5,
    alignItems: 'center',
    backgroundColor: '#007bff',
  },
  buttonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 18,
  },
  loadingContainer: {
    paddingTop: 20,
  },
  resultText: {
    marginTop: 20,
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
  },
});

export default App;

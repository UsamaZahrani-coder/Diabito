declare module "expo-image-picker" {
  export interface ImagePickerResult {
    canceled: boolean;
    assets: Array<{
      uri: string;
      width: number;
      height: number;
      type?: string;
      base64?: string;
    }>;
  }

  export interface ImagePickerOptions {
    mediaTypes: MediaTypeOptions;
    allowsEditing?: boolean;
    aspect?: [number, number];
    quality?: number;
    base64?: boolean;
    exif?: boolean;
  }

  export enum MediaTypeOptions {
    Images = "Images",
    Videos = "Videos",
    All = "All"
  }

  export function launchImageLibraryAsync(options?: ImagePickerOptions): Promise<ImagePickerResult>;
  export function launchCameraAsync(options?: ImagePickerOptions): Promise<ImagePickerResult>;
  export function requestMediaLibraryPermissionsAsync(): Promise<{ status: string }>;
  export function requestCameraPermissionsAsync(): Promise<{ status: string }>;
}